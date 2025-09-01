from torch.utils.data import Dataset
import os
import torch
import numpy as np
import imageio 

class LightfieldVideoDataset(Dataset):
    def __init__(self, basedir, grid_size, train_frames_num, start_frames_num):
        view_dirs = sorted([d for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d)) and '_' in d])
        self.view_dirs = view_dirs
        self.grid_size = grid_size
        self.train_frames_num = train_frames_num
        self.start_frames_num = start_frames_num
        self.frames = []
    

        for view_dir in view_dirs:
            view_path = os.path.join(basedir, view_dir)
            frames = sorted([f for f in os.listdir(view_path) if f.endswith('.png')])
            if len(frames) < start_frames_num + train_frames_num:
                raise ValueError(f"{view_dir}에서 start_frames_num({start_frames_num})부터 train_frames_num({train_frames_num})까지의 프레임이 부족합니다.")

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

    def get_uvst(self, x, y):
        aspect = self.W / self.H
        uv_scale = 1
        st_scale = 0.1
        
        y = 2.0 * (y / (self.grid_size - 1)) - 1
        x = 2.0 * (x / (self.grid_size - 1)) - 1

        u = np.linspace(-1, 1, self.W, dtype=np.float32)
        v = np.linspace(1, -1, self.H, dtype=np.float32) / aspect
        vu = list(np.meshgrid(u, v))
        
        u = vu[0] * uv_scale
        v = vu[1] * uv_scale
        
        s = np.ones_like(vu[0]) * y * st_scale
        t = np.ones_like(vu[1]) * x * st_scale
        
        uvst = np.stack([u, v, s, t], axis=-1)
        
        return uvst
    
    def get_norm_xy(self, x, y):
        x = 2.0 * (x / (self.grid_size - 1)) - 1
        y = 2.0 * (y / (self.grid_size - 1)) - 1
        x *= 0.1
        y *= 0.1
        return x, y

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        file_path = self.frames[idx]
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)

        x, y = map(int, folder_name.split('_'))
        x, y = self.get_norm_xy(x, y)
        uvst = self.get_uvst(x, y)
        cur_time = int(os.path.splitext(file_name)[0]) / 10.0

        img = imageio.imread(file_path)
        img = (np.array(img) / 255.).astype(np.float32)
        uvst = (np.array(uvst)).astype(np.float32)
        cur_time = (np.array(cur_time)).astype(np.float32)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        uvst = torch.from_numpy(uvst)
        cur_time = torch.tensor(cur_time, dtype=torch.float32)
        img = torch.from_numpy(img)

        data = {}
        data['x'] = x
        data['y'] = y
        data['uvst'] = uvst
        data['cur_time'] = cur_time
        data['img'] = img

        return data
    
    def get_render_uvst_time(self):
        render_uvsts = []
        for i in range(self.train_frames_num):
            render_uvst = self.get_uvst(0, 0)
            render_uvsts.append(render_uvst)
        render_uvsts = np.array(render_uvsts).astype(np.float32)

        render_time = np.linspace(0., (self.train_frames_num - 1) / 10.0, self.train_frames_num)
        return render_uvsts, render_time