from utils.base_utils import *
from .utils.video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
import torchvision.transforms as T
from torchvision import transforms
import pandas as pd
import os
import re
from .utils.egoexobench import *
import torch

FAIL_MSG = 'Failed to obtain answer via API.'

class EgoExoBench_MCQ(VideoBaseDataset):
    MD5 = 'b5c3cfe5d316f1a67a4076991f16ca9c'
    TYPE = 'Video_MCQ'

    def __init__(self, dataset='EgoExoBench_MCQ'):
        super().__init__(dataset=dataset, nframe=64)
        self.frame_fps = 2

    @classmethod
    def supported_datasets(cls):
        return ['EgoExoBench_MCQ']

    def prepare_dataset(self, dataset_name='EgoExoBench_MCQ'):

        self.video_root = osp.join(LMUDataRoot(),'videos')


        data_file = osp.join(LMUDataRoot(),'annotations',  f'{dataset_name}.tsv')
        # transform
        self.transform = T.Compose([
            Stack(),
            ToTorchFormatTensor()
        ])
        
        return dict(root=LMUDataRoot(), data_file=data_file)
    
    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=16):
        start, end = bound if bound else (-100000, 100000)
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = (end_idx - start_idx) / num_segments
        mid_seg_size = seg_size / 2
        indices = np.arange(num_segments)
        frame_indices = start_idx + mid_seg_size + np.round(seg_size * indices)
        return frame_indices.astype(int)
    

    def load_into_video_and_process(self, media, mcq_idx):
        try:
            from moviepy import VideoFileClip, ImageSequenceClip
        except:
            pass
        video_root = self.video_root

        if media['type'] in ['image']:
            original_image_path = osp.join(video_root, media['image_paths'][0])
            processed_video_path = osp.join(video_root, 'processed_video', f'{mcq_idx}.jpg')
            if not os.path.exists(processed_video_path):
                shutil.copy(original_image_path, processed_video_path)
            return dict(type='image', value=processed_video_path)  
        
        elif media['type'] in ['frames']:
            input_images = [osp.join(video_root, im) for im in media['image_paths']]
            processed_video_path = osp.join(video_root, 'processed_video', f'{mcq_idx}.mp4')
            media['nframes'] = len(input_images)//2*2
            if not os.path.exists(processed_video_path):
                # using MoviePy to transform images into mp4
                image_files = sorted(input_images)
                image_clip = ImageSequenceClip(image_files, fps=self.frame_fps)
                image_clip.write_videofile(processed_video_path, codec='libx264')
                image_clip.close()
        elif media['type'] in ['video']:
            original_video_path = osp.join(video_root, media['video_path'])
            processed_video_path = osp.join(video_root, 'processed_video', f'{mcq_idx}.mp4')
            if 'video_start' in media and 'video_end' in media and media['video_start'] is not None and media['video_end'] is not None:
                video_start, video_end = media['video_start'], media['video_end']                    
                if not os.path.exists(processed_video_path):                
                    video_clip = VideoFileClip(original_video_path)
                    clip = video_clip.subclipped(video_start, min(video_end, video_clip.duration))
                    clip.write_videofile(processed_video_path)
                    clip.close()
            else:
                if not os.path.exists(processed_video_path):
                    shutil.copy(original_video_path, processed_video_path)
        else:
            raise ValueError(f"Unsupported media type: {media['type']}")

        return dict(type='video', value=processed_video_path, nframes=media.get('nframes', 8))
    
    def save_video_into_images(self, media, mcq_idx):
        bound = None
        video_root = self.video_root
        if os.path.exists(osp.join(video_root, 'processed_frames', str(mcq_idx))):
            dir_name = osp.join(video_root, 'processed_frames', str(mcq_idx))
            return [osp.join(dir_name,img_file) for img_file in os.listdir(dir_name)]
        
        if media['type'] in ['frames', 'image']:
            media_paths = [osp.join(video_root, im) for im in media['image_paths']]
            save_dir = osp.join(video_root, 'processed_frames', str(mcq_idx))
            os.makedirs(save_dir, exist_ok=True)
            input_images = []
            for media_path in media_paths:
                img_path = media_path.split('/')[-1]
                save_image_path = osp.join(save_dir, img_path)
                if not os.path.exists(save_image_path):
                    shutil.copy(media_path, save_image_path)
                input_images.append(save_image_path)
                
            return input_images
            
        if 'video_start' in media and 'video_end' in media and media['video_start'] is not None and media['video_end'] is not None:
            bound = (
                media['video_start'], media['video_end']
            )
        video_path = os.path.join(video_root, media['video_path'])
        
        def read_video(video_path, bound=None, num_segments=16):
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            max_frame = len(vr) - 1
            fps = float(vr.get_avg_fps())

            images_group = list()
            frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
            save_dir = osp.join(video_root, 'processed_frames', str(mcq_idx))
            
            if osp.exists(save_dir) and len(os.listdir(save_dir)) > 0:
                return None, frame_indices
        
            for frame_index in frame_indices:
                img = Image.fromarray(vr[frame_index].asnumpy())
                images_group.append(img)
            torch_imgs = self.transform(images_group)
            return torch_imgs, frame_indices

        def save_video_frames(imgs, video_root, frame_indices, mcq_idx):

            save_dir = osp.join(video_root, 'processed_frames', str(mcq_idx))
            os.makedirs(save_dir, exist_ok=True)
            frame_paths = [osp.join(save_dir, f'{fidx:07d}.jpg') for fidx in frame_indices]

            flag = np.all([osp.exists(pth) for pth in frame_paths])

            if not flag:
                block_size = imgs.size(0) // len(frame_indices)
                split_tensors = torch.split(imgs, block_size)
                to_pil = transforms.ToPILImage()
                images = [to_pil(arr) for arr in split_tensors]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)

            return frame_paths
    
        
        torch_imgs, frame_indices = read_video(video_path, bound, media['nframes'])
        img_frame_paths = save_video_frames(torch_imgs, video_root, frame_indices, mcq_idx)
        return img_frame_paths
    
    def process_text_and_media(self, text, media_list, video_llm, mcq_idx):

        message = []
        chunks = re.split(r'(<image>|<video>)', text)
        media_index = 0
        media_list = eval(media_list)

        placeholder_count = sum(1 for chunk in chunks if chunk in ['<image>', '<video>'])
        assert placeholder_count == len(media_list), \
            f"Placeholder count {placeholder_count} does not match media list length {len(media_list)}."
            

        for chunk in chunks:
            if chunk in ['<image>', '<video>']:
                if video_llm:
                    media_content = self.load_into_video_and_process(media_list[media_index], f'question{mcq_idx}_video{media_index}')
                    message.append(media_content)
                else:
                    # Save the video as individual image frames for processing
                    img_frame_paths = self.save_video_into_images(media_list[media_index], f'question{mcq_idx}_video{media_index}')
                    for im in img_frame_paths:
                        message.append(dict(type='image', value=im))
                    
                media_index += 1
            elif chunk.strip():
                message.append(dict(type='text', value=chunk.strip()))

        return message
    
    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            mcq_idx = line
            assert line < len(self)
            line = self.data.iloc[line]
            mcq_idx = int(line['index'])
        text = line['question'] + '\nOptions:\n' + line['options'] + '\n' + line['response_format']
        print(mcq_idx)
        message = self.process_text_and_media(text, line['medias'], video_llm, mcq_idx)
        return message
    
    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.egoexobench import get_dimension_rating, extract_characters_regex, extract_option

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')
            assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']

            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]

                if extract_characters_regex(pred) == '':
                    extract_pred = extract_option(
                        model,
                        data.loc[data['index'] == idx].to_dict(orient='records')[0],
                        'EgoExoBench_MCQ',
                    )
                    data.loc[idx, 'score'] = int(extract_pred == ans)
                else:
                    data.loc[idx, 'score'] = int(extract_characters_regex(pred) == ans)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
    