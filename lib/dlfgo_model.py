import os, time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import vggrid

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from submodules.gridencoder import GridEncoder

'''Twoplane Model'''
class DLFGO_twoplane(torch.nn.Module):
    def __init__(self, ray_min, ray_max, grid_num, ray_type='twoplane',
                 grid_dim=16, mlp_depth=2, mlp_width=128, pe=0,
                 decomp='4d', levels=[1],):
        super(DLFGO_twoplane, self).__init__()
        self.register_buffer('ray_min', torch.Tensor(ray_min))
        self.register_buffer('ray_max', torch.Tensor(ray_max))
        self.grid_num = grid_num
        self._set_grid_resolution(grid_num)
        self.grid_dim = grid_dim
        self.mlp_kwargs = {
            'mlp_depth': mlp_depth, 'mlp_width': mlp_width,
        }
        self.pe = pe
        self.decomp = decomp
        self.levels = levels
        if self.levels == 1:
            self.level = [1]
        elif self.levels == 2:
            self.level = [1, 2]
        elif self.levels == 3:
            self.level = [2, 4, 8]
        elif self.levels == 4:
            self.level = [2, 4, 6, 8]
        elif self.levels == 5:
            self.level = [1, 2, 3, 4, 5]
        elif self.levels == 8:
            self.level = [1, 2, 3, 4, 5, 6, 7, 8]
        elif self.levels == 16:
            self.level = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            
        if self.decomp == 'all':
            self.grid_combinations_2d = [
                [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
            ]
            self.grid_combinations_3d = [
                [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
            ]
            self.grid_combinations_4d = [
                [0, 1, 2, 3],
            ]
            self.grid_2d = nn.ModuleList()
            self.grid_3d = nn.ModuleList()
            self.grid_4d = nn.ModuleList()
            for i, level in enumerate(self.level):
                self.grid_2d.append(nn.ModuleList([
                    vggrid.create_grid(
                        type='LF2DGrid', channels=self.grid_dim,
                    ray_min=self.ray_min[grid_comb], ray_max=self.ray_max[grid_comb],
                    grid_size=torch.tensor(self.grid_size[grid_comb] * level, dtype=torch.long)
                ) for grid_comb in self.grid_combinations_2d
                ]))
                self.grid_3d.append(nn.ModuleList([
                    vggrid.create_grid(
                        type='LF3DGrid', channels=self.grid_dim,
                        ray_min=self.ray_min[grid_comb], ray_max=self.ray_max[grid_comb],
                        grid_size=torch.tensor(self.grid_size[grid_comb] * level, dtype=torch.long)
                    ) for grid_comb in self.grid_combinations_3d
                ]))
                self.grid_4d.append(nn.ModuleList([
                    vggrid.create_grid(
                        type='LF4DGrid', channels=self.grid_dim,
                        ray_min=self.ray_min[grid_comb], ray_max=self.ray_max[grid_comb],
                        grid_size=torch.tensor(self.grid_size[grid_comb] * level, dtype=torch.long)
                    ) for grid_comb in self.grid_combinations_4d
                ]))
            mlp_dim = self.grid_dim * len(self.grid_combinations_2d) * self.levels + self.grid_dim * len(self.grid_combinations_3d) * self.levels + self.grid_dim * len(self.grid_combinations_4d) * self.levels
            print('dlfgo_twoplane: grid', self.grid_2d, self.grid_3d, self.grid_4d)
            
        else:
            if self.decomp == '1d':
                self.grid_type = 'LF1DGrid'
                self.grid_combinations = [
                    [0], [1], [2], [3],
                ]
            elif self.decomp == '2d':
                self.grid_type = 'LF2DGrid'
                self.grid_combinations = [
                    [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
                    # [0, 1], [2, 3],
                ]
                # self.grid_type2 = 'LF4DGrid'
                # self.grid_combinations2 = [
                #     [0, 1, 2, 3],
                # ]
            elif self.decomp == '3d':
                self.grid_type = 'LF3DGrid'
                self.grid_combinations = [
                    [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
                ]
            elif self.decomp == '4d':
                self.grid_type = 'LF4DGrid'
                self.grid_combinations = [
                    [0, 1, 2, 3],
                ]
            self.grid = nn.ModuleList()
            for i, level in enumerate(self.level):
                grid_modules = nn.ModuleList([
                    vggrid.create_grid(
                        type=self.grid_type, channels=self.grid_dim,
                        ray_min=self.ray_min[grid_comb], ray_max=self.ray_max[grid_comb],
                        grid_size=torch.tensor(self.grid_size[grid_comb] * level, dtype=torch.long)
                    ) for grid_comb in self.grid_combinations
                ])
                self.grid.append(grid_modules)
            mlp_dim = self.grid_dim * len(self.grid_combinations) * self.levels
            print('dlfgo_twoplane: grid', self.grid)
            
        if self.pe > 0:
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(self.pe)]))
            mlp_dim += (2+2*pe*2)
        # self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(self.pe)]))
        # mlp_dim += (3+3*pe*2)
        # self.encoder = GridEncoder(input_dim=4, num_levels=8, level_dim=2, base_resolution=16, log2_hashmap_size=28, desired_resolution=512, gridtype='hash', align_corners=True)
        # print('dlfgo_twoplane: hash encoder', self.encoder)
        # mlp_dim = self.encoder.output_dim
        # self.attn_xyuv = nn.Sequential(nn.Linear(4, 32), nn.ReLU(inplace=True), nn.Linear(32, 2), nn.Sigmoid())
        # self.attn_xy = nn.Sequential(nn.Linear(4, 32), nn.ReLU(inplace=True), nn.Linear(32, self.grid_dim), nn.Sigmoid())
        # self.attn_uv = nn.Sequential(nn.Linear(4, 32), nn.ReLU(inplace=True), nn.Linear(32, self.grid_dim), nn.Sigmoid())
        # # print('dlfgo_twoplane: attn_xyuv', self.attn_xyuv)
        # print('dlfgo_twoplane: attn_xy, attn_uv', self.attn_xy, self.attn_uv)
        # self.mlp_input = nn.Sequential(
        #     nn.Linear(4, 32), nn.ReLU(inplace=True),
        #     nn.Linear(32, 4),
        # )
        # nn.init.constant_(self.mlp_input[-1].bias, 0)
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim, mlp_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(mlp_width, mlp_width), nn.ReLU(inplace=True))
                for _ in range(mlp_depth-2)
            ],
            nn.Linear(mlp_width, 3),
        )
        nn.init.constant_(self.mlp[-1].bias, 0)
        print('dlfgo_twoplane: mlp', self.mlp)
            
    def _set_grid_resolution(self, grid_num):
        self.grid_size = torch.tensor(grid_num, dtype=torch.int)
        print('dlfgo_twoplane: grid_size ', self.grid_size)

    def get_kwargs(self):
        return {
            'ray_min': self.ray_min.cpu().numpy(),
            'ray_max': self.ray_max.cpu().numpy(),
            'grid_num': self.grid_num,
            'grid_dim': self.grid_dim,
            **self.mlp_kwargs,
            'pe': self.pe,
            'decomp': self.decomp,
            'levels': self.levels,
        }
    
    # @torch.no_grad()
    # def scale_volume_grid(self, factor=1.2):
    #     print('dlfgo_twoplane: scale_volume_grid start')
    #     ori_grid_size = self.grid_size.tolist()
    #     new_grid_size = self.grid_size * factor
    #     self._set_grid_resolution(new_grid_size)
    #     print('dlfgo_twoplane: scale_volume_grid scale world_size from', ori_grid_size, 'to', self.grid_size)
    #     for i in range(self.levels):
    #         for j, grid_comb in enumerate(self.grid_combinations):
    #             self.grid[i][j].scale_volume_grid(new_grid_size[grid_comb])
                
    def total_variation_add_grad(self, weight, dense_mode=True):
        for i in range(self.levels):
            for j, grid_comb in enumerate(self.grid_combinations):
                w = weight * self.grid_size.max() / 128
                self.grid[i][j].total_variation_add_grad(w, w, w, w, dense_mode)
        
    def forward(self, ray, global_step=None):

        # ray [130172, 4]
        results = []
        if self.decomp == 'all':
            for i in range(self.levels):
                for j, grid_comb in enumerate(self.grid_combinations_2d):
                    results.append(self.grid_2d[i][j](ray[:, grid_comb]))
                for j, grid_comb in enumerate(self.grid_combinations_3d):
                    results.append(self.grid_3d[i][j](ray[:, grid_comb]))
                for j, grid_comb in enumerate(self.grid_combinations_4d):
                    results.append(self.grid_4d[i][j](ray[:, grid_comb]))
        else:
            for i in range(self.levels):
                if self.decomp == '2d':
                    # f_4d = self.grid[-1][0](ray) # [130172, 6]
                    # for j, grid_comb in enumerate(self.grid_combinations):
                    #     results.append(self.grid[i][j](ray[:, grid_comb]) + f_4d[:, j].unsqueeze(-1))
                    # results.append(self.grid[-1][0](ray))
                    for j, grid_comb in enumerate(self.grid_combinations):
                        results.append(self.grid[i][j](ray[:, grid_comb]))
                else:
                    for j, grid_comb in enumerate(self.grid_combinations):
                        results.append(self.grid[i][j](ray[:, grid_comb]))
        # results_xy = []
        # results_uv = []
        # if global_step >= 10000:
        # alpha = 1e-04
        # embedded_ray = self.mlp_input(ray)
        # embedded_ray = torch.tanh(embedded_ray)
        # embedded_ray = embedded_ray * 2 - 1
        # ray = ray + embedded_ray * alpha
        # embedded_ray = embedded_ray * 2 - 1
        # ray = ray + embedded_ray
        # for i in range(self.levels):
        #     # f_4d = self.grid[-self.levels+i](ray)
        #     for j, grid_comb in enumerate(self.grid_combinations):
        #         f_grid = self.grid[i][j](ray[:, grid_comb])
        #         results.append(f_grid)
                # results.append(torch.cat([f_grid, f_4d[:, j].unsqueeze(-1)], dim=-1))
                # results.append(f_grid + f_4d[:, j].unsqueeze(-1))
        # results.append(self.grid[-1](ray))
                # if grid_comb == [0, 1]:
                #     results_xy.append(f_grid)
                # elif grid_comb == [2, 3]:
                #     results_uv.append(f_grid)
        
        results = torch.cat(results, dim=-1)
        
        
        # self.grid_time[t](ray)
        
            
        # results = torch.stack(results, dim=-1)
        # results = torch.sum(results, dim=-1)
        # results = self.encoder(ray)
        # results_xy = torch.cat(results_xy, dim=-1)
        # results_uv = torch.cat(results_uv, dim=-1)
        # attn_xy = self.attn_xy(ray)
        # attn_uv = self.attn_uv(ray)
        # results_xy = results_xy * attn_xy
        # results_uv = results_uv * attn_uv
        # results = torch.cat([results_xy, results_uv], dim=-1)
        if self.pe > 0:
            xy_emb = (ray[:, :2].unsqueeze(-1) * self.viewfreq).flatten(-2)
            xy_emb = torch.cat([ray[:, :2], xy_emb.sin(), xy_emb.cos()], -1)
            results = torch.cat([xy_emb, results], dim=-1)
            
        results = self.mlp(results)
        render_result = torch.sigmoid(results)
        return render_result