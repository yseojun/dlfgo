import os, time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import vggrid

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from submodules.gridencoder import GridEncoder

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
def get_embedder(multires, input_dims, i=0):
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

'''Twoplane Model'''
class DLFGO_twoplane_mlp(torch.nn.Module):
    def __init__(self, ray_type='twoplane', mlp_depth=5, mlp_width=256, pe=0,
                 multires=10, input_dims=5):
        super(DLFGO_twoplane_mlp, self).__init__()
        self.embed_fn, self.embed_dim = get_embedder(multires, input_dims, 0)
        self.multires = multires
        self.input_dims = input_dims

        self.mlp_kwargs = {
            'mlp_depth': mlp_depth, 'mlp_width': mlp_width,
        }
        self.pe = pe
        
        self.skips = [4]
        
        # Skip connection을 위해 Sequential 대신 ModuleList 사용
        self.mlp_layers = nn.ModuleList()
        
        # 첫 번째 레이어
        self.mlp_layers.append(nn.Linear(self.embed_dim, mlp_width))
        
        # 중간 레이어들
        for i in range(mlp_depth):
            if i in self.skips:
                # Skip connection이 있는 레이어는 입력 차원 추가
                self.mlp_layers.append(nn.Linear(mlp_width + self.embed_dim, mlp_width))
            else:
                self.mlp_layers.append(nn.Linear(mlp_width, mlp_width))
        
        # 출력 레이어
        self.mlp_layers.append(nn.Linear(mlp_width, 3))
        
        self.relu = nn.ReLU(inplace=True)
        
        # nn.init.constant_(self.mlp[-1].bias, 0)
        print('dlfgo_twoplane: mlp', self.mlp_layers)
            
    def get_kwargs(self):
        return {
            **self.mlp_kwargs,
            'pe': self.pe,
            'multires': self.multires,
            'input_dims': self.input_dims,
        }
                
    def total_variation_add_grad(self, weight, dense_mode=True):
        for i in range(self.levels):
            for j, grid_comb in enumerate(self.grid_combinations):
                w = weight * self.grid_size.max() / 128
                self.grid[i][j].total_variation_add_grad(w, w, w, w, dense_mode)
        
    def forward(self, ray, global_step=None):
        # breakpoint()
        embedded_ray = self.embed_fn(ray)
        
        # Forward pass through MLP with skip connections
        x = embedded_ray
        for i, layer in enumerate(self.mlp_layers):
            # 빌드 시 중간 레이어 인덱스(i_mid = i-1)를 기준으로 스킵 적용
            if i > 0 and i < len(self.mlp_layers) - 1 and (i - 1) in self.skips:
                # Skip connection: 원본 embedded_ray를 concatenate
                x = torch.cat([x, embedded_ray], dim=-1)
            
            if i < len(self.mlp_layers) - 1:
                x = self.relu(layer(x))
            else:
                x = layer(x)  # 마지막 레이어는 활성화 함수 없음
        
        results = x
        render_result = torch.sigmoid(results)
        return render_result