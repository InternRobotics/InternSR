import numpy as np
from pathlib import Path
from PIL import Image
import os
import torch
from torch import Tensor
import clip
from typing import Tuple, Union
import mmengine
from tqdm import tqdm
import cv2
from argparse import ArgumentParser
from multiprocessing import Pool
import multiprocessing as mp



global INPUT_FOLDER
global OUTPUT_FOLDER
def points_img2cam(
        points: Union[Tensor, np.ndarray],
        cam2img: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Project points in image coordinates to camera coordinates.

    Args:
        points (Tensor or np.ndarray): 2.5D points in 2D images with shape
            [N, 3], 3 corresponds with x, y in the image and depth.
        cam2img (Tensor or np.ndarray): Camera intrinsic matrix. The shape can
            be [3, 3], [3, 4] or [4, 4].

    Returns:
        Tensor or np.ndarray: Points in 3D space with shape [N, 3], 3
        corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = torch.tensor(points[:, 2]).view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D

def inpaint_depth(depth):
    """
    inpaints depth using opencv
    Input: torch tensor with depthvalues: H, W
    Output: torch tensor with depthvalues: H, W
    """
    depth_inpaint = cv2.inpaint(depth, (depth == 0).astype(np.uint8), 5, cv2.INPAINT_NS)
    depth[depth == 0] = depth_inpaint[depth == 0]
    return depth

def points_cam2img(points_3d: Union[Tensor, np.ndarray],
                   proj_mat: Union[Tensor, np.ndarray],
                   with_depth: bool = False) -> Union[Tensor, np.ndarray]:
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (Tensor or np.ndarray): Points in shape (N, 3).
        proj_mat (Tensor or np.ndarray): Transformation matrix between
            coordinates.
        with_depth (bool): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        Tensor or np.ndarray: Points in image coordinates with shape [N, 2] if
        ``with_depth=False``, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, \
        'The dimension of the projection matrix should be 2 ' \
        f'instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or \
        (d1 == 4 and d2 == 4), 'The shape of the projection matrix ' \
        f'({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(4,
                                      device=proj_mat.device,
                                      dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res


def extract_embodiedscan_frames(video):

    image_path = Path(video)
    video_frames = sorted(image_path.glob("*.jpg"))  # find all the color frames of the rgbd video
    frames = [str(video_frame) for video_frame in video_frames]

    images = []
    depths = []
    poses = []

    if 'scannet' in frames[0]:
        video = frames[0].split('/')[-4] + '/' + frames[0].split('/')[-2]
    elif '3rscan' in frames[0]:
        video = frames[0].split('/')[-4] + '/' + frames[0].split('/')[-3]

    if video.split('/')[0]+'/'+MAPPING[video.split('/')[1]] not in SCENE:
        return None

    video_info = SCENE[video.split('/')[0]+'/'+MAPPING[video.split('/')[1]]]

    for frame in frames:
        path = Path(frame)
        frame_name = str(Path(*path.parts[-4:]))
        pose = np.array(video_info[frame_name]['pose']) # 4x4 array
        image = frame
        if 'scannet' in frame:
            depth = frame.replace('jpg', 'png')
        elif '3rscan' in frame:
            depth = frame.replace('color.jpg', 'depth.pgm')
        else:
            raise NotImplementedError
        # we need to ensure that the frame has valid pose
        # pose = frame.replace('color', 'pose').replace('png', 'txt')
        images.append(image)
        depths.append(depth)
        poses.append(pose)
    depth_intrinsic_file = np.array(video_info['depth_intrinsic'])  # 4x4 array
    intrinsic_file = np.array(video_info['intrinsic']) # 4x4 array
    axis_align_matrix_file = np.array(video_info['axis_align_matrix'])  # 4x4 array
    video_info = dict()
    video_info['sample_image_files'] = images
    video_info['sample_depth_image_files'] = depths
    video_info['sample_pose_files'] = poses
    video_info['depth_intrinsic_file'] = depth_intrinsic_file
    video_info['intrinsic_file'] = intrinsic_file
    video_info['axis_align_matrix_file'] = axis_align_matrix_file

    return video_info


def create_3rscan_depth_image(depth_img, depth_intrinsic, intrinsic, image_size):

    depth_img = np.array(depth_img)
    # print('depth_image shape:', depth_img.shape)
    ws = np.arange(depth_img.shape[1])
    hs = np.arange(depth_img.shape[0])
    us, vs = np.meshgrid(ws, hs)
    grid = np.stack(
        [us.astype(np.float32),
            vs.astype(np.float32), depth_img], axis=-1).reshape(-1, 3)
    nonzero_indices = depth_img.reshape(-1).nonzero()[0]
    grid3d = points_img2cam(torch.tensor(grid), torch.tensor(depth_intrinsic))
    # print('grid3d shape:', grid3d.shape)
    # print('depth_intrinsic:', depth_intrinsic)
    points = grid3d[nonzero_indices]
    # print('points shape:', points.shape)

    # fatch feature
    width, height = image_size
    # print('color image shape:', width, height)
    points2d = points_cam2img(points, intrinsic, True)
    # print('points2d:', points2d[:20])
    points2d = np.round(points2d)
    # print('intrinsic:', intrinsic)
    # print(points2d.shape, points2d[:20])

    # print(points2d[:10])
    depth_img = np.full((height, width), np.inf)

    # 遍历每个点并更新深度图
    for x, y, z in points2d:
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            if depth_img[y, x] > z:  # 如果当前位置的深度值大于新点的深度值
                depth_img[y, x] = z

    depth_img[np.isinf(depth_img)] = 0
    depth_img[depth_img>3000] = 0
    depth_img = inpaint_depth(depth_img.astype(np.float32))
    depth_img = depth_img.astype(np.uint16)
    # print(np.max(depth_img))
    depth_img = Image.fromarray(depth_img)  # 和 rgb 图同尺寸

    return depth_img

SCENE = mmengine.load('./data/annotations/embodiedscan_infos_full.json')
MAPPING = mmengine.load('./data/annotations/3rscan_mapping.json')


def count_files_with_extension(directory, extension):
    return sum(1 for file in os.listdir(directory) if file.endswith(extension))

def compare_image_counts(path1, path2):
    jpg_count_path1 = count_files_with_extension(path1, '.jpg')

    png_count_path2 = count_files_with_extension(path2, '.png')

    return jpg_count_path1 == png_count_path2

def process(video):
    global INPUT_FOLDER
    global OUTPUT_FOLDER
    video = os.path.join(INPUT_FOLDER, video, 'sequence')
    if os.path.exists(video.replace(INPUT_FOLDER, OUTPUT_FOLDER)) and compare_image_counts(video, video.replace(INPUT_FOLDER, OUTPUT_FOLDER)):
        return
    else:
        os.makedirs(video.replace(INPUT_FOLDER, OUTPUT_FOLDER),exist_ok=True)
        video_info = extract_embodiedscan_frames(video)
        if video_info is None:
            return
        depth_intrinsic = video_info['depth_intrinsic_file']
        if not isinstance(depth_intrinsic, np.ndarray):
            depth_intrinsic = np.loadtxt(depth_intrinsic)

        intrinsic = video_info['intrinsic_file']
        if not isinstance(intrinsic, np.ndarray):
            intrinsic = np.loadtxt(intrinsic)

        for id, image_file in enumerate(tqdm(video_info['sample_image_files'])):
            image = Image.open(image_file).convert('RGB')  
            image_size = image.size
            # image = self.image_processor.preprocess(images=image, do_rescale=do_rescale, do_normalize=do_normalize, return_tensors=return_tensors)['pixel_values'][0] # [3, H, W]
            depth_image = Image.open(video_info['sample_depth_image_files'][id])
            depth_image_size = depth_image.size
            # print('depth_image_size:', depth_image_size)
            # print('depth_iamge size:', depth_image.size)

            depth_image = create_3rscan_depth_image(depth_image, depth_intrinsic, intrinsic, image_size)
            depth_file = image_file.replace('color.jpg', 'depth.png').replace(INPUT_FOLDER, OUTPUT_FOLDER)
            depth_path = os.path.dirname(depth_file)

            if not os.path.exists(depth_path):
                os.makedirs(depth_path)
            print('save in: ',depth_file)
            # print(image_file, np.array(depth_image)[20:30, 20:30])
            # print(depth_file)
            # print(depth_file)
            # depth_image.save(os.path.join(OUTPUT_FOLDER, video, save_depth, depth_file))
            depth_image.save(depth_file)


def list_subdirectories(directory):
    # 列出目录下的所有子目录
    subdirs = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return subdirs

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--input_folder',
                        required=True)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()

    NUM_PROCESSES = 32
    INPUT_FOLDER = args.input_folder
    OUTPUT_FOLDER = args.output_folder
    
    with Pool(NUM_PROCESSES) as p:
        list(tqdm(p.imap(process, list_subdirectories(args.input_folder)), total=len(list_subdirectories(args.input_folder))))