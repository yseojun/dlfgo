import os
import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F
import time

from torch.autograd import gradcheck

import grid_interpole

def create_grid(type, **kwargs):
    if type == 'LF2DGrid':
        return LF2DGrid(**kwargs)
    elif type == 'LF3DGrid':
        return LF3DGrid(**kwargs)
    elif type == 'LF4DGrid':
        return LF4DGrid(**kwargs)
    elif type == 'LF5DGrid':
        return LF5DGrid(**kwargs)
    elif type == 'LF1DGrid':
        return LF1DGrid(**kwargs)
    else:
        raise NotImplementedError
    
''' LF 1D grid
'''
class LF1DGrid(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, **kwargs):
        super(LF1DGrid, self).__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.register_buffer('ray_min', torch.Tensor(ray_min))
        self.register_buffer('ray_max', torch.Tensor(ray_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, grid_size]))
        
    def forward(self, ray):
        # pytorch 기본함수 구현
        ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min))
        grid_indices = ind_norm * (self.grid_size - 1)
        out = self.interpolate_1d_flatten(grid_indices)
        
        # CUDA 구현
        # ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min)) * 2 - 1
        # out = grid_interpole.grid_interpolate_1d(self.grid, ind_norm)
        return out
    
    def interpolate_1d_flatten(self, grid_indices):
        """
        grid_indices: (N,) 실수 좌표
        self.grid: (C, U)
        반환값: (N, C)
        """
        _, C, U = self.grid.shape
        N = grid_indices.shape[0]
        
        # 1) bottom/top 정수 인덱스, weight 계산
        bottom = grid_indices.floor().long()      # (N,)
        top    = bottom + 1
        w      = (grid_indices - bottom.float())  # (N,)
        omw    = 1.0 - w                          # (N,)
        
        # 2) strides, flatten
        s0 = 1
        P = U
        grid_flat = self.grid.view(C, P)         # (C, P)
        
        # 3) 2개 조합에 대한 mask와 선형 인덱스 계산
        b0, t0 = bottom, top
        
        coords = [
            (b0, t0),
        ]
        
        masks = []
        idxs  = []
        for (i0,i1) in coords:
            valid = (
                (i0 >= 0) & (i0 < U) &
                (i1 >= 0) & (i1 < U)
            )  # (N,)
            lin   = i0*s0
            idxs.append(lin)
            masks.append(valid)
            
        # (2, N) 텐서로 쌓기
        idxs = torch.stack(idxs, dim=0)
        masks = torch.stack(masks, dim=0).float()  # 1.0 or 0.0
        
        # 4) clamp + gather
        idxs_clamped = idxs.clamp(0, P-1)            # OOB 방지
        flat_idxs = idxs_clamped.contiguous().view(-1).long()
        vals_flat = torch.index_select(grid_flat, 1, flat_idxs)
        vals = vals_flat.view(C, 2, N).permute(1, 0, 2)
        
        # 5) weight 계산
        comb_w = torch.stack([
            omw[:,0],
            w  [:,0],
        ], dim=0)  # (2, N)
        
        # 6) mask 적용 & 최종 합산
        weighted = vals * comb_w.unsqueeze(1) * masks.unsqueeze(1)  # (2, C, N)
        out = weighted.sum(dim=0).transpose(0,1)                   # (N, C)
        
        return out
    
    def extra_repr(self):
        return f'channels={self.channels}, grid_size={self.grid_size.tolist()}'

''' LF 2D grid
'''
class LF2DGrid(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, **kwargs):
        super(LF2DGrid, self).__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.register_buffer('ray_min', torch.Tensor(ray_min))
        self.register_buffer('ray_max', torch.Tensor(ray_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *grid_size]))
        
    def forward(self, ray):
        # pytorch 기본함수 구현
        # ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min))
        # grid_indices = (ind_norm * (self.grid_size - 1))
        # out = self.interpolate_2d_flatten(grid_indices)
        
        # torch.nn.functional.grid_sample 구현
        shape = ray.shape[:-1]
        ray = ray.reshape(1,1,-1,2)
        ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        return out
    
    def interpolate_2d_flatten(self, grid_indices):
        """
        grid_indices: (N,2) 실수 좌표
        self.grid: (C, H, W)
        반환값: (N, C)
        """
        _, C, D0, D1 = self.grid.shape
        N = grid_indices.shape[0]
        
        # 1) bottom/top 정수 인덱스, weight 계산
        bottom = grid_indices.floor().long()      # (N,2)
        top    = bottom + 1
        w      = (grid_indices - bottom.float())  # (N,2)
        omw    = 1.0 - w                          # (N,2)
        
        # 2) strides, flatten
        s0, s1 = D1, 1
        P = D0 * D1
        grid_flat = self.grid.view(C, P)         # (C, P)
        
        # 3) 4개 조합에 대한 mask와 선형 인덱스 계산
        b0, b1 = bottom.unbind(1)
        t0, t1 = top   .unbind(1)
        
        coords = [
            (b0, b1),
            (t0, b1),
            (b0, t1),
            (t0, t1),
        ]
        
        masks = []
        idxs  = []
        for (i0,i1) in coords:
            valid = (
                (i0 >= 0) & (i0 < D0) &
                (i1 >= 0) & (i1 < D1)
            )  # (N,)
            lin   = i0*s0 + i1*s1
            idxs.append(lin)
            masks.append(valid)
            
        # (4, N) 텐서로 쌓기
        idxs = torch.stack(idxs, dim=0)
        masks = torch.stack(masks, dim=0).float()  # 1.0 or 0.0
        
        # 4) clamp + gather
        idxs_clamped = idxs.clamp(0, P-1)            # OOB 방지
        flat_idxs = idxs_clamped.contiguous().view(-1).long()
        vals_flat = torch.index_select(grid_flat, 1, flat_idxs)
        vals = vals_flat.view(C, 4, N).permute(1, 0, 2)
        
        # 5) weight 계산
        comb_w = torch.stack([
            omw[:,0]*omw[:,1],
            w  [:,0]*omw[:,1],
            omw[:,0]*w  [:,1],
            w  [:,0]*w  [:,1],
        ], dim=0)  # (4, N)
        
        # 6) mask 적용 & 최종 합산
        weighted = vals * comb_w.unsqueeze(1) * masks.unsqueeze(1)  # (4, C, N)
        out = weighted.sum(dim=0).transpose(0,1)                   # (N, C)
        
        return out
    
    def total_variation_add_grad(self, wx, wy, wz, wt, dense_mode=True):
        return grid_interpole.total_variation_add_grad(self.grid, self.grid.grad, wx, wy, 0.0, 0.0, 2, dense_mode)
    
    def extra_repr(self):
        return f'channels={self.channels}, grid_size={self.grid_size.tolist()}'

''' LF 3D grid
'''
class LF3DGrid(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, **kwargs):
        super(LF3DGrid, self).__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.register_buffer('ray_min', torch.Tensor(ray_min))
        self.register_buffer('ray_max', torch.Tensor(ray_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *grid_size]))
        
    def forward(self, ray):
        # pytorch 기본함수 구현
        # ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min))
        # grid_indices = (ind_norm * (self.grid_size - 1))
        # out = self.interpolate_3d_flatten(grid_indices)
        
        # torch.nn.functional.grid_sample 구현
        shape = ray.shape[:-1]
        ray = ray.reshape(1,1,1,-1,3) # [1, 1, 1, N, 3]
        ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        
        # CUDA 구현
        # ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min)) * 2 - 1
        # out = grid_interpole.grid_interpolate_3d(self.grid[0], ind_norm)
        return out
    
    def interpolate_3d_flatten(self, grid_indices):
        """
        grid_indices: (N,3) 실수 좌표
        self.grid: (C, D0, D1, D2)
        반환값: (N, C)
        """
        _, C, D0, D1, D2 = self.grid.shape
        N = grid_indices.shape[0]
        
        # 1) bottom/top 정수 인덱스, weight 계산
        bottom = grid_indices.floor().long()      # (N,3)
        top    = bottom + 1
        w      = (grid_indices - bottom.float())  # (N,3)
        omw    = 1.0 - w                          # (N,3)
        
        # 2) strides, flatten
        s0, s1, s2 = D1*D2, D2, 1
        P = D0 * D1 * D2
        grid_flat = self.grid.view(C, P)         # (C, P)
        
        # 3) 8개 조합에 대한 mask와 선형 인덱스 계산
        b0, b1, b2 = bottom.unbind(1)
        t0, t1, t2 = top   .unbind(1)
        
        # 리스트로 묶어서 for 없이 한꺼번에 계산
        coords = [
            (b0, b1, b2),
            (t0, b1, b2),
            (b0, t1, b2),
            (t0, t1, b2),
            (b0, b1, t2),
            (t0, b1, t2),
            (b0, t1, t2),
            (t0, t1, t2),
        ]

        # 각 조합에 대해 valid mask와 idxs 생성
        masks = []
        idxs  = []
        for (i0,i1,i2) in coords:
            valid = (
                (i0 >= 0) & (i0 < D0) &
                (i1 >= 0) & (i1 < D1) &
                (i2 >= 0) & (i2 < D2)
            )  # (N,)
            lin   = i0*s0 + i1*s1 + i2*s2
            idxs.append(lin)
            masks.append(valid)
            
        # (8, N) 텐서로 쌓기
        idxs = torch.stack(idxs, dim=0)
        masks = torch.stack(masks, dim=0).float()  # 1.0 or 0.0
        
        # 4) clamp + gather
        idxs_clamped = idxs.clamp(0, P-1)            # OOB 방지
        flat_idxs = idxs_clamped.contiguous().view(-1).long()
        vals_flat = torch.index_select(grid_flat, 1, flat_idxs)
        vals = vals_flat.view(C, 8, N).permute(1, 0, 2)
        
        # 5) weight 계산
        comb_w = torch.stack([
            omw[:,0]*omw[:,1]*omw[:,2],
            w  [:,0]*omw[:,1]*omw[:,2],
            omw[:,0]*w  [:,1]*omw[:,2],
            w  [:,0]*w  [:,1]*omw[:,2],
            omw[:,0]*omw[:,1]*w  [:,2],
            w  [:,0]*omw[:,1]*w  [:,2],
            omw[:,0]*w  [:,1]*w  [:,2],
            w  [:,0]*w  [:,1]*w  [:,2],
        ], dim=0)  # (8, N)
        
        # 6) mask 적용 & 최종 합산
        weighted = vals * comb_w.unsqueeze(1) * masks.unsqueeze(1)  # (8, C, N)
        out = weighted.sum(dim=0).transpose(0,1)                   # (N, C)
        
        return out
    
    def total_variation_add_grad(self, wx, wy, wz, wt, dense_mode=True):
        return grid_interpole.total_variation_add_grad(self.grid, self.grid.grad, wx, wy, wz, 0.0, 3, dense_mode)

    def extra_repr(self):
        return f'channels={self.channels}, grid_size={self.grid_size.tolist()}'

''' LF 4D grid
'''
class LF4DGrid(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, **kwargs):
        super(LF4DGrid, self).__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.register_buffer('ray_min', torch.Tensor(ray_min))
        self.register_buffer('ray_max', torch.Tensor(ray_max))
        self.grid = nn.Parameter(torch.zeros([channels, *grid_size]).requires_grad_(True))
        
    def forward(self, ray):
        # pytorch 기본함수 구현
        # ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min))
        # grid_indices = (ind_norm * (self.grid_size - 1))
        # out = self.interpolate_4d_flatten(grid_indices)
        
        # CUDA 구현
        ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min)) * 2 - 1
        out = grid_interpole.grid_interpolate_4d(self.grid, ind_norm)
        return out
    
    def interpolate_4d_flatten(self, grid_indices):
        """
        grid_indices: (N,4) 실수 좌표
        self.grid: (C, D0, D1, D2, D3)
        반환값: (N, C)
        """
        C, D0, D1, D2, D3 = self.grid.shape
        N = grid_indices.shape[0]

        # 1) bottom/top 정수 인덱스, weight 계산
        bottom = grid_indices.floor().long()      # (N,4)
        top    = bottom + 1
        w      = (grid_indices - bottom.float())  # (N,4)
        omw    = 1.0 - w                          # (N,4)

        # 2) strides, flatten
        s0, s1, s2, s3 = D1*D2*D3, D2*D3, D3, 1
        P = D0 * D1 * D2 * D3
        grid_flat = self.grid.view(C, P)         # (C, P)

        # 3) 16개 조합에 대한 mask와 선형 인덱스 계산
        b0, b1, b2, b3 = bottom.unbind(1)
        t0, t1, t2, t3 = top   .unbind(1)

        # 리스트로 묶어서 for 없이 한꺼번에 계산
        coords = [
            (b0, b1, b2, b3),
            (t0, b1, b2, b3),
            (b0, t1, b2, b3),
            (t0, t1, b2, b3),
            (b0, b1, t2, b3),
            (t0, b1, t2, b3),
            (b0, t1, t2, b3),
            (t0, t1, t2, b3),
            (b0, b1, b2, t3),
            (t0, b1, b2, t3),
            (b0, t1, b2, t3),
            (t0, t1, b2, t3),
            (b0, b1, t2, t3),
            (t0, b1, t2, t3),
            (b0, t1, t2, t3),
            (t0, t1, t2, t3),
        ]
        # 각 조합에 대해 valid mask와 idxs 생성
        masks = []
        idxs  = []
        for (i0,i1,i2,i3) in coords:
            valid = (
                (i0 >= 0) & (i0 < D0) &
                (i1 >= 0) & (i1 < D1) &
                (i2 >= 0) & (i2 < D2) &
                (i3 >= 0) & (i3 < D3)
            )  # (N,)
            lin   = i0*s0 + i1*s1 + i2*s2 + i3*s3
            idxs.append(lin)
            masks.append(valid)
        
        # (16, N) 텐서로 쌓기
        idxs = torch.stack(idxs, dim=0)
        masks = torch.stack(masks, dim=0).float()  # 1.0 or 0.0

        # 4) clamp + gather
        idxs_clamped = idxs.clamp(0, P-1)            # OOB 방지
        flat_idxs = idxs_clamped.contiguous().view(-1).long()
        vals_flat = torch.index_select(grid_flat, 1, flat_idxs)
        vals = vals_flat.view(C, 16, N).permute(1, 0, 2)

        # 5) weight 계산
        comb_w = torch.stack([
            omw[:,0]*omw[:,1]*omw[:,2]*omw[:,3],
            w  [:,0]*omw[:,1]*omw[:,2]*omw[:,3],
            omw[:,0]*w  [:,1]*omw[:,2]*omw[:,3],
            w  [:,0]*w  [:,1]*omw[:,2]*omw[:,3],
            omw[:,0]*omw[:,1]*w  [:,2]*omw[:,3],
            w  [:,0]*omw[:,1]*w  [:,2]*omw[:,3],
            omw[:,0]*w  [:,1]*w  [:,2]*omw[:,3],
            w  [:,0]*w  [:,1]*w  [:,2]*omw[:,3],
            omw[:,0]*omw[:,1]*omw[:,2]*w  [:,3],
            w  [:,0]*omw[:,1]*omw[:,2]*w  [:,3],
            omw[:,0]*w  [:,1]*omw[:,2]*w  [:,3],
            w  [:,0]*w  [:,1]*omw[:,2]*w  [:,3],
            omw[:,0]*omw[:,1]*w  [:,2]*w  [:,3],
            w  [:,0]*omw[:,1]*w  [:,2]*w  [:,3],
            omw[:,0]*w  [:,1]*w  [:,2]*w  [:,3],
            w  [:,0]*w  [:,1]*w  [:,2]*w  [:,3],
        ], dim=0)  # (16, N)

        # 6) mask 적용 & 최종 합산
        weighted = vals * comb_w.unsqueeze(1) * masks.unsqueeze(1)  # (16, C, N)
        out = weighted.sum(dim=0).transpose(0,1)                   # (N, C)

        return out
    
    def total_variation_add_grad(self, wx, wy, wz, wt, dense_mode=True):
        return grid_interpole.total_variation_add_grad(self.grid, self.grid.grad, wx, wy, wz, wt, 4, dense_mode)
            
    def extra_repr(self):
        return f'channels={self.channels}, grid_size={self.grid_size.tolist()}'

''' LF 5D grid
'''
class LF5DGrid(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, **kwargs):
        super(LF5DGrid, self).__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.grid = nn.Parameter(torch.zeros([1, channels, *grid_size]))
        self.register_buffer('ray_min', torch.Tensor(ray_min))
        self.register_buffer('ray_max', torch.Tensor(ray_max))
        
    def forward(self, ray):
        # pytorch 구현
        ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min))
        grid_indices = (ind_norm * (self.grid_size - 1))
        out = self.interpolate_5d_flatten(grid_indices)
        return out
    
    def interpolate_5d_flatten(self, grid_indices):
        """
        grid_indices: (N,5) 실수 좌표
        self.grid: (C, D0, D1, D2, D3, D4)
        반환값: (N, C)
        """
        _, C, D0, D1, D2, D3, D4 = self.grid.shape
        N = grid_indices.shape[0]
        
        # 1) bottom/top 정수 인덱스, weight 계산
        bottom = grid_indices.floor().long()      # (N,5)
        top    = bottom + 1
        w      = (grid_indices - bottom.float())  # (N,5)
        omw    = 1.0 - w                          # (N,5)
        
        # 2) strides, flatten
        s0, s1, s2, s3, s4 = D1*D2*D3*D4, D2*D3*D4, D3*D4, D4, 1
        P = D0 * D1 * D2 * D3 * D4
        grid_flat = self.grid.view(C, P)         # (C, P)
        
        # 3) 32개 조합에 대한 mask와 선형 인덱스 계산
        b0, b1, b2, b3, b4 = bottom.unbind(1)
        t0, t1, t2, t3, t4 = top   .unbind(1)
        
        # 리스트로 묶어서 for 없이 한꺼번에 계산
        coords = [
            (b0, b1, b2, b3, b4),
            (t0, b1, b2, b3, b4),
            (b0, t1, b2, b3, b4),
            (t0, t1, b2, b3, b4),
            
        ]
        
        # 각 조합에 대해 valid mask와 idxs 생성
        masks = []
        idxs  = []
        for (i0,i1,i2,i3,i4) in coords:
            valid = (
                (i0 >= 0) & (i0 < D0) &
                (i1 >= 0) & (i1 < D1) &
                (i2 >= 0) & (i2 < D2) &
                (i3 >= 0) & (i3 < D3) &
                (i4 >= 0) & (i4 < D4)
            )  # (N,)
            lin   = i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4
            idxs.append(lin)
            masks.append(valid)
            
        # (32, N) 텐서로 쌓기
        idxs = torch.stack(idxs, dim=0)
        masks = torch.stack(masks, dim=0).float()  # 1.0 or 0.0
        
        # 4) clamp + gather
        idxs_clamped = idxs.clamp(0, P-1)            # OOB 방지
        vals = (grid_flat
                .unsqueeze(0)                       # (1, C, P)
                .expand(32, C, P)                  # (32, C, P)
                .gather(2,
                        idxs_clamped
                        .unsqueeze(1)              # (32, 1, N)
                        .expand(-1, C, -1)         # (32, C, N)
                    )
            )  # (32, C, N)
        
        # 5) weight 계산
        comb_w = torch.stack([
            omw[:,0]*omw[:,1]*omw[:,2]*omw[:,3]*omw[:,4],
            w  [:,0]*omw[:,1]*omw[:,2]*omw[:,3]*omw[:,4],
            omw[:,0]*w  [:,1]*omw[:,2]*omw[:,3]*omw[:,4],
            w  [:,0]*w  [:,1]*omw[:,2]*omw[:,3]*omw[:,4],
        ], dim=0)  # (32, N)
        
        # 6) mask 적용 & 최종 합산
        weighted = vals * comb_w.unsqueeze(1) * masks.unsqueeze(1)  # (32, C, N)
        out = weighted.sum(dim=0).transpose(0,1)                   # (N, C)

        return out
            
    def extra_repr(self):
        return f'channels={self.channels}, lf_world_size={self.lf_world_size.tolist()}'
