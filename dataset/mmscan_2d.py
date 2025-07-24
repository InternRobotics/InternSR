import os
import cv2
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from .utils.image_base import ImageBaseDataset
from utils.base_utils import *
from utils import track_progress_rich

def sample_idx(length, num_samples=24):

    if length <= num_samples:
        return list(range(length))

    return np.linspace(0, length - 1, num=num_samples, dtype=int)

def draw_box3d_points_on_img(
    img,
    point,
    label_name,
    extrinsic_c2w,
    intrinsic,
):
    """
    Draw a 3D box on an image.
    Args:
        img (numpy.ndarray): shape (h, w, 3)
        box (open3d.geometry.OrientedBoundingBox): A 3D box.
        color (tuple): RGB color of the box.
        label (str): Label of the box.
        extrinsic_c2w (numpy.ndarray): 4x4 extrinsic, camera to world.
        intrinsic (numpy.ndarray): 4x4 (extended) intrinsic.
        alpha (float): Alpha value of the drawn faces.
        occupency_map (numpy.ndarray): boolean array, occupency map of the image.
    Returns:
        img (numpy.ndarray): Updated image with the box drawn on it.
        occupency_map (numpy.ndarray): updated occupency map
    """
    
    extrinsic = np.linalg.inv(extrinsic_c2w)
    h, w, _ = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    center = point
    center_2d = (
        intrinsic
        @ extrinsic
        @ np.array([center[0], center[1], center[2], 1]).reshape(4, 1)
    )
    center_2d = center_2d[:2] / center_2d[2]
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 1
    text_color = (0, 0, 255)  
    background_color = (255, 255, 255)  

    (text_width, text_height), _ = cv2.getTextSize(label_name, font, font_scale, font_thickness)
    
    position = (int(center_2d[0]), int(center_2d[1]))


    padding = 2
    rect_start = (position[0] - padding, position[1] + padding)
    rect_end = (position[0] + text_width + padding, position[1] - text_height - padding)
    
    b_img = img


    cv2.rectangle(b_img, rect_start, rect_end, background_color, -1)
    cv2.putText(b_img, label_name, position, font, font_scale, text_color, font_thickness)
    return b_img

class ImageILDataset(ImageBaseDataset):

    DATASET_URL = {'mmscan_2d': ''}
    DATASET_MD5 = {'mmscan_2d': ''}

    TYPE = 'IL'
    
    def __init__(self, dataset='MMBench', skip_noimg=True, ratio = 0.01, num_images = 24):
        ROOT = LMUDataRoot()
        # You can override this variable to save image files to a different directory
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, 'images', 'mmscan')
        self.meta_only = True
        llava_form_dir = osp.join(ROOT, 'annotations')
        if not os.path.exists(os.path.join(llava_form_dir,f'mmscan_qa_val_{ratio}.json')):
            assert "MMScan only support ratio for 0.01 / 0.05 / 0.1 / 0.5 / 1.0."
        raw_data = json.load(open(os.path.join(llava_form_dir,f'mmscan_qa_val_{ratio}.json')))
        meta_dir = osp.join(ROOT, 'annotations/embodiedscan_video_meta')
        self.raw_file = f'annotation_images/mmscan_qa_val_{ratio}'
        self.meat_dir_dict = {}
        for meta_file in os.listdir(meta_dir):
            self.meat_dir_dict[meta_file.split('.')[0]] = json.load(open(os.path.join(meta_dir,meta_file)))
        self.num_images = num_images
        self.data = self.process_raw_json(raw_data)

    def process_raw_json(self, raw_data):
        data = {
            "index": [],
            "question": [],
            "answer": [],
            "ID": [],
            "image_path":[]
        }
        os.makedirs(os.path.join(self.img_root,self.raw_file+'_'+str(self.num_images)),exist_ok=True)
        index = 0
        print("Processing the 2d modality data for MMScan-2D")
        for sample in tqdm(raw_data):
            index+=1
            video_id = sample['video']
            prompt_id = sample['prompt_id'].replace('/','-')
            question = sample['conversations'][0]['value'].replace("<video>\n",'')
            if '<boxes>' in question:
                ori_question = question.split('<boxes>. ')[1]
                num_box = len(sample["target"]["boxes"])
                question = f'The {num_box} objects involved in the question have been marked on the images (labeled with red numbers {list(range(1,num_box+1))}). The labels on the images indicate the projected center position of each corresponding object in every image. '+\
                'The object corresponding to a label may be occluded by other objects; please determine which object each label refers to by combining all images. '+ ori_question
            question = 'Answer the qusetion based on the images of the scene. '+ question
            answer = str(sample['conversations'][1]['value'])
            image_paths = []
               

            axis_align_matrix = np.array(self.meat_dir_dict['axis_align_matrix'][video_id])
            images = self.meat_dir_dict['image'][video_id]
            depths = self.meat_dir_dict['depth'][video_id]
            intrinsics = np.array(self.meat_dir_dict['intrinsic'][video_id])
            pose = np.array(self.meat_dir_dict['pose'][video_id])
            extrinsics_c2w = np.matmul(axis_align_matrix, pose)
            ids = sample_idx(len(images))
            question = '<ImageHere>'*len(ids)+question
            
            sample_image_dir = os.path.join(self.img_root,self.raw_file+'_'+str(self.num_images),prompt_id )
            if os.path.exists(sample_image_dir) and len(os.listdir(sample_image_dir))==len(ids):
                image_paths = [os.path.join(sample_image_dir,file) for file in os.listdir(sample_image_dir)]
            else:
                os.makedirs(os.path.join(self.img_root,self.raw_file+'_'+str(self.num_images),prompt_id ),exist_ok=True)
                for idx in ids:

                    image = cv2.imread(os.path.join(self.img_root,images[idx]))
                    cnt = 1
                    if 'target' in sample:
                        for boxes in sample["target"]["boxes"]:
                            box_center = np.array(boxes[:3])
                            image = draw_box3d_points_on_img(image,box_center,str(cnt),extrinsics_c2w[idx],intrinsics[idx])
                            cnt+=1
                    save_path = os.path.join(self.img_root,self.raw_file+'_'+str(self.num_images),prompt_id,f'{idx}.jpg')
                    cv2.imwrite(save_path,image)
                    image_paths.append(save_path)
            data['index'].append(index)
            data['question'].append(question)
            data['answer'].append(answer)
            data['image_path'].append(str(image_paths))
            data['ID'].append(prompt_id)
        data = pd.DataFrame(data)
        return data
        
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
  
        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)
  
        q = line['question']

        pics_number = 0
        if '<ImageHere>' in q:
            content = []
            tag_number = q.count('<ImageHere>')
            images = tgt_path[pics_number: pics_number + tag_number]
            pics_number += tag_number
            q_split = q.split('<ImageHere>')
            for i in range(tag_number):
                qsp, im = q_split[i], images[i]
                if qsp != '':
                    content.append(dict(type='text', value=qsp))
                content.append(dict(type='image', value=im))
            if q_split[-1] != '':
                content.append(dict(type='text', value=q_split[-1]))
        else:
            content = [dict(type='text', value=q)]

        return content

class MMScanDataset(ImageILDataset):

    def evaluate(self, eval_file, **judge_kwargs):
        print('******************************************')
        print(f'The result is save in {eval_file}, use MMScan evaluator to evaluate the prediction.')
        print('******************************************')

        return {}