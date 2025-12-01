import torch.utils.data as data
import torch
import numpy as np
from src.config import cfg
import os
import json
import imageio
import cv2


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        """
        Description:
            __init__ 函数负责从磁盘中 load 指定格式的文件，计算并存储为特定形式

        Input:
            @kwargs: 读取的参数
        Output:
            None
        """
        super(Dataset, self).__init__()
        
        data_root = kwargs['data_root']
        split = kwargs['split']
        scene = cfg.scene
        self.split = split
        self.data_root = os.path.join(data_root, scene)
        self.input_ratio = kwargs.get('input_ratio', 1.0)
        self.batch_size = cfg.task_arg.N_rays
        
        transform_file = os.path.join(self.data_root, f'transforms_{split}.json')
        with open(transform_file, 'r') as f:
            meta = json.load(f)
        
        self.camera_angle_x = meta['camera_angle_x']
        
        images_list = []
        poses_list = []
        
        frames = meta['frames']
        
        # Filter frames based on cams parameter
        if 'cams' in kwargs:
            cams = kwargs['cams']
            selected_frames = []
            for cam_idx in cams:
                if cam_idx == -1:  # use all
                    selected_frames = frames
                    break
                elif 0 <= cam_idx < len(frames):
                    selected_frames.append(frames[cam_idx])
            frames = selected_frames
        
        # Skip frames for test dataset
        test_skip = cfg.task_arg.get('test_skip', 1)
        if split == 'test' and test_skip > 1:
            frames = frames[::test_skip]
        
        for frame in frames:
            image_path = os.path.join(self.data_root, frame['file_path'][2:] + '.png')
            if not os.path.exists(image_path):
                image_path = os.path.join(self.data_root, frame['file_path'] + '.png')
            
            img = imageio.imread(image_path)
            img = img.astype(np.float32) / 255.0
            
            # Handle alpha channel
            if img.shape[-1] == 4:
                img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
            
            # Resize
            if self.input_ratio != 1.0:
                img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, 
                               interpolation=cv2.INTER_AREA)
            
            images_list.append(torch.from_numpy(img))
            poses_list.append(torch.from_numpy(np.array(frame['transform_matrix'], dtype=np.float32)))
        
        self.images = torch.stack(images_list)  # (N, H, W, 3)
        self.poses = torch.stack(poses_list)    # (N, 4, 4)
        
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        
        self.focal = 0.5 * self.W / np.tan(0.5 * self.camera_angle_x)
        
        # Pre-compute all rays for training
        if split == 'train' and cfg.task_arg.get('no_batching', True):
            all_rays = []
            all_rgbs = []
            
            for i in range(len(self.images)):
                rays_o, rays_d = self.get_rays(self.H, self.W, self.focal, self.poses[i])
                rays = torch.cat([rays_o, rays_d], dim=-1)  # (H, W, 6)
                all_rays.append(rays.reshape(-1, 6))
                all_rgbs.append(self.images[i].reshape(-1, 3))
            
            self.all_rays = torch.cat(all_rays, dim=0)  # (N*H*W, 6)
            self.all_rgbs = torch.cat(all_rgbs, dim=0)  # (N*H*W, 3)
    
    def get_rays(self, H, W, focal, c2w):
        """
        Generate rays for a given camera pose.
        
        Args:
            H: image height
            W: image width
            focal: focal length
            c2w: camera to world transformation matrix (4, 4) - torch.Tensor
        
        Returns:
            rays_o: ray origins (H, W, 3)
            rays_d: ray directions (H, W, 3)
        """
        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32), 
            torch.arange(H, dtype=torch.float32),
            indexing='xy'
        )
        
        dirs = torch.stack([
            (i - W * 0.5) / focal, 
            -(j - H * 0.5) / focal, 
            -torch.ones_like(i)
        ], dim=-1)  # (H, W, 3)
        
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)  # (H, W, 3)
        rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
        
        return rays_o, rays_d

    def __getitem__(self, index):
        """
        Description:
            __getitem__ 函数负责在运行时提供给网络一次训练需要的输入，以及 ground truth 的输出
        对 NeRF 来说，分别是 1024 条光线以及 1024 个 RGB值

        Input:
            @index: 图像下标, 范围为 [0, len-1]
        Output:
            @ret: 包含所需数据的字典
        """
        if self.split == 'train':
            if cfg.task_arg.get('no_batching', True):
                # Random sample rays from all images
                select_inds = torch.randperm(len(self.all_rays))[:self.batch_size]
                rays = self.all_rays[select_inds]  # (N_rays, 6)
                rgb = self.all_rgbs[select_inds]   # (N_rays, 3)
            else:
                # Sample rays from a single image
                img = self.images[index]
                pose = self.poses[index]
                
                rays_o, rays_d = self.get_rays(self.H, self.W, self.focal, pose)
                rays = torch.cat([rays_o, rays_d], dim=-1)  # (H, W, 6)
                rays = rays.reshape(-1, 6)  # (H*W, 6)
                rgb = img.reshape(-1, 3)    # (H*W, 3)
                
                # Random sample
                select_inds = torch.randperm(len(rays))[:self.batch_size]
                rays = rays[select_inds]
                rgb = rgb[select_inds]
            
            ret = {
                'rays': rays,
                'rgb': rgb
            }
        else:
            # For testing, return all rays for the image
            img = self.images[index]
            pose = self.poses[index]
            
            rays_o, rays_d = self.get_rays(self.H, self.W, self.focal, pose)
            rays = torch.cat([rays_o, rays_d], dim=-1)  # (H, W, 6)
            rays = rays.reshape(-1, 6)  # (H*W, 6)
            rgb = img.reshape(-1, 3)    # (H*W, 3)
            
            ret = {
                'rays': rays,
                'rgb': rgb,
                'meta': {'H': self.H, 'W': self.W}
            }
        
        return ret

    def __len__(self):
        """
        Description:
            __len__ 函数返回训练或者测试的数量

        Input:
            None
        Output:
            @len: 训练或者测试的数量
        """
        if self.split == 'train' and cfg.task_arg.get('no_batching', True):
            return cfg.ep_iter
        else:
            return len(self.images)
