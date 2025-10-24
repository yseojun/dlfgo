import os
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

class LightFieldVideoDataset(Dataset):
    def __init__(self, basedir, grid_size_x, grid_size_y, train_frames_num, start_frames_num, sample_frame_num=None):
        view_dirs = sorted([d for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d)) and '_' in d])
        self.view_dirs = view_dirs
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.train_frames_num = train_frames_num
        self.start_frames_num = start_frames_num
        self.sample_frame_num = sample_frame_num
        self.frames = []

        for view_dir in view_dirs:
            view_path = os.path.join(basedir, view_dir)
            frames = sorted([f for f in os.listdir(view_path) if f.endswith('.png')])
            if len(frames) < start_frames_num + train_frames_num:
                raise ValueError(f"{view_dir}에서 start_frames_num({start_frames_num})부터 train_frames_num({train_frames_num})까지의 프레임이 부족합니다.")

        print(f"| init Dataset")
        print(f"| num_frames: {len(frames)}")
        print(f"| start_frames: {start_frames_num}")
        print(f"| train_frames: {train_frames_num}")

        for view_dir in view_dirs:
            view_path = os.path.join(basedir, view_dir)
            frames = sorted([f for f in os.listdir(view_path) if f.endswith('.png')])
            frames = frames[start_frames_num:start_frames_num + train_frames_num]
            frames_path = [os.path.join(view_path, frame) for frame in frames]
            self.frames.extend(frames_path)

        self.H, self.W = imageio.imread(self.frames[0]).shape[:2]
        print(f"| H: {self.H}, W: {self.W}")

    def get_uvst(self, grid_x, grid_y):
        """
        Generate uvst coordinates for a given grid position.
        
        Args:
            grid_x: Grid x index (0 to grid_size_x-1)
            grid_y: Grid y index (0 to grid_size_y-1)
        """
        aspect = self.W / self.H
        uv_scale = 1
        st_scale = 0.25
        
        # Normalize grid coordinates to [-1, 1]
        norm_y = 2.0 * (grid_y / (self.grid_size_y - 1)) - 1
        norm_x = 2.0 * (grid_x / (self.grid_size_x - 1)) - 1

        u = np.linspace(-1, 1, self.W, dtype=np.float32)
        v = np.linspace(1, -1, self.H, dtype=np.float32) / aspect
        vu = list(np.meshgrid(u, v))
        
        u = vu[0] * uv_scale
        v = vu[1] * uv_scale
        
        s = np.ones_like(vu[0]) * norm_y * st_scale
        t = np.ones_like(vu[1]) * norm_x * st_scale
        
        uvst = np.stack([s, t, u, v], axis=-1)
        
        return uvst
    
    def get_norm_xy(self, grid_x, grid_y):
        """
        Normalize grid coordinates to [-0.1, 0.1] range.
        
        Args:
            grid_x: Grid x index (0 to grid_size_x-1)
            grid_y: Grid y index (0 to grid_size_y-1)
        """
        norm_x = 2.0 * (grid_x / (self.grid_size_x - 1)) - 1
        norm_y = 2.0 * (grid_y / (self.grid_size_y - 1)) - 1
        norm_x *= 0.1
        norm_y *= 0.1
        return norm_x, norm_y

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        file_path = self.frames[idx]
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)

        # Parse grid coordinates from folder name (e.g., "00_01" -> y=0, x=1)
        grid_y, grid_x = map(int, folder_name.split('_'))
        
        # Get normalized coordinates for pose/camera position
        norm_x, norm_y = self.get_norm_xy(grid_x, grid_y)
        
        # Get uvst light field coordinates (uses original grid indices)
        uvst = self.get_uvst(grid_x, grid_y)
        
        # Extract frame number from filename (supports both "123.png" and "frame_00123.png" formats)
        file_name_no_ext = os.path.splitext(file_name)[0]
        if file_name_no_ext.startswith('frame_'):
            frame_num = int(file_name_no_ext.split('_')[1])
        else:
            frame_num = int(file_name_no_ext)
        
        # Calculate time based on frame number
        # Assuming frame numbering starts from 1 or 0, normalize to [0, 1] range
        cur_time = (frame_num - self.start_frames_num) / max(1, self.train_frames_num - 1)

        img = imageio.imread(file_path)
        img = (np.array(img) / 255.).astype(np.float32)
        uvst = (np.array(uvst)).astype(np.float32)
        cur_time = (np.array(cur_time)).astype(np.float32)

        norm_x = torch.tensor(norm_x, dtype=torch.float32)
        norm_y = torch.tensor(norm_y, dtype=torch.float32)
        uvst = torch.from_numpy(uvst)
        cur_time = torch.tensor(cur_time, dtype=torch.float32)
        img = torch.from_numpy(img)

        data = {}
        data['x'] = norm_x
        data['y'] = norm_y
        data['uvst'] = uvst
        data['cur_time'] = cur_time
        data['img'] = img

        return data
    
    def get_render_uvst_time(self):
        """
        Get render uvst and time coordinates for all training frames.
        Uses center grid position (grid_size_x//2, grid_size_y//2).
        """
        render_uvsts = []
        center_x = self.grid_size_x // 2
        center_y = self.grid_size_y // 2
        
        for i in range(self.train_frames_num):
            render_uvst = self.get_uvst(center_x, center_y)
            render_uvsts.append(render_uvst)
        render_uvsts = np.array(render_uvsts).astype(np.float32)

        # Time normalized to [0, 1] range matching __getitem__ logic
        render_time = np.linspace(0., 1.0, self.train_frames_num).astype(np.float32)
        return render_uvsts, render_time
    
