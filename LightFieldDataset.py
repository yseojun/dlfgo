import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from lib.utils import get_rays_of_a_view_twoplane

class LightFieldDataSet(Dataset):
    def __init__(self, data_dict, indices, ray_type='twoplane', pixels_per_image=1024, load2gpu_on_the_fly=False):
        """
        이미지 단위 샘플링을 위한 Light Field Dataset
        
        Args:
            data_dict: load_data()에서 반환된 데이터 딕셔너리
            indices: 사용할 이미지 인덱스 (i_train, i_val, i_test)
            ray_type: 'twoplane' 또는 'plucker'
            pixels_per_image: 각 이미지에서 샘플링할 픽셀 수
            load2gpu_on_the_fly: 메모리 효율을 위한 온디맨드 로딩
        """
        self.images = data_dict['images']
        self.poses = data_dict['poses'] 
        self.HW = data_dict['HW']
        self.Ks = data_dict['Ks']
        self.indices = indices
        self.ray_type = ray_type
        self.pixels_per_image = pixels_per_image
        self.load2gpu_on_the_fly = load2gpu_on_the_fly
        
        # 이미지 크기 정보
        self.H, self.W = self.HW[0]  # 첫 번째 이미지 크기 (모든 이미지가 같다고 가정)
        
        # 각 이미지의 총 픽셀 수
        self.total_pixels_per_image = self.H * self.W

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        하나의 이미지에서 pixels_per_image 개수만큼 픽셀을 랜덤 샘플링
        
        Returns:
            dict: {
                'rgb': [pixels_per_image, 3],
                'rays': [pixels_per_image, 4] for twoplane,
                'img_idx': int
            }
        """
        img_idx = self.indices[idx]
        
        # 이미지와 관련 정보 로드
        if isinstance(self.images, list):
            # irregular_shape인 경우
            image = self.images[img_idx]  # [H, W, 3]
        else:
            # regular shape인 경우
            image = self.images[img_idx]  # [H, W, 3]
        
        pose = self.poses[img_idx]  # [3, 4]
        H, W = self.HW[img_idx]
        K = self.Ks[img_idx]  # [3, 3]
        
        # Ray 생성 (실시간 계산)
        if self.ray_type == 'twoplane':
            rays = get_rays_of_a_view_twoplane(H, W, K, pose)  # [H*W, 4]
        else:
            raise NotImplementedError(f'Ray type {self.ray_type} not implemented')
        
        # 이미지를 flatten
        rgb_flat = image.view(-1, 3)  # [H*W, 3]
        
        # 픽셀 랜덤 샘플링 (CPU 디바이스 확실히 보장)
        if self.pixels_per_image < self.total_pixels_per_image:
            # 랜덤 인덱스 생성 (CPU에서)
            perm = torch.randperm(self.total_pixels_per_image, device='cpu')[:self.pixels_per_image]
            # 모든 텐서를 CPU로 이동 후 인덱싱
            rgb_flat_cpu = rgb_flat.cpu()
            rays_cpu = rays.cpu()
            rgb_sampled = rgb_flat_cpu[perm]  # [pixels_per_image, 3]
            rays_sampled = rays_cpu[perm]     # [pixels_per_image, 4]
        else:
            # 모든 픽셀 사용
            rgb_sampled = rgb_flat.cpu()
            rays_sampled = rays.cpu()
            
        return {
            'rgb': rgb_sampled,
            'rays': rays_sampled,
            'img_idx': img_idx
        }


def create_lightfield_dataset(data_dict, split='train', batch_size=8192, num_images_per_batch=8):
    """
    Light Field Dataset 생성 헬퍼 함수
    
    Args:
        data_dict: load_data()에서 반환된 데이터 딕셔너리
        split: 'train', 'val', 'test' 중 하나
        batch_size: 총 배치 크기 (픽셀 단위)
        num_images_per_batch: 배치당 이미지 수
        
    Returns:
        dataset, dataloader
    """
    # split에 따른 인덱스 선택
    if split == 'train':
        indices = data_dict['i_train']
    elif split == 'val':
        indices = data_dict['i_val'] 
    elif split == 'test':
        indices = data_dict['i_test']
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # 이미지당 픽셀 수 계산
    pixels_per_image = batch_size // num_images_per_batch
    
    # Dataset 생성
    dataset = LightFieldDataSet(
        data_dict=data_dict,
        indices=indices,
        ray_type='twoplane',  # 기본값
        pixels_per_image=pixels_per_image
    )
    
    # DataLoader 생성 
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """
        배치를 하나의 텐서로 합치는 함수
        """
        rgb_list = [item['rgb'] for item in batch]
        rays_list = [item['rays'] for item in batch]
        img_idx_list = [item['img_idx'] for item in batch]
        
        # 모든 샘플을 연결
        rgb_batch = torch.cat(rgb_list, dim=0)    # [batch_size, 3]
        rays_batch = torch.cat(rays_list, dim=0)  # [batch_size, 4]
        
        return {
            'rgb': rgb_batch,
            'rays': rays_batch,
            'img_indices': img_idx_list
        }
    
    # 테스트를 위해 shuffle 비활성화 (CUDA generator 문제 회피)
    dataloader = DataLoader(
        dataset,
        batch_size=num_images_per_batch,
        shuffle=False,  # 임시로 비활성화
        collate_fn=collate_fn,
        num_workers=0,  # 이미지가 이미 메모리에 있으므로
        pin_memory=False  # CUDA 호환성을 위해 False로 설정
    )
    
    return dataset, dataloader
