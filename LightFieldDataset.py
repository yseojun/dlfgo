import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from lib.utils import get_rays_of_a_view_twoplane

class LightFieldDataSet(Dataset):
    def __init__(self, data_dict, indices, ray_type='twoplane', batch_size=8192, load2gpu_on_the_fly=False):
        """
        모든 ray를 한 epoch에 학습하기 위한 Light Field Dataset
        
        Args:
            data_dict: load_data()에서 반환된 데이터 딕셔너리
            indices: 사용할 이미지 인덱스 (i_train, i_val, i_test)
            ray_type: 'twoplane' 또는 'plucker'
            batch_size: 배치당 ray 수
            load2gpu_on_the_fly: 메모리 효율을 위한 온디맨드 로딩
        """
        self.images = data_dict['images']
        self.poses = data_dict['poses'] 
        self.HW = data_dict['HW']
        self.Ks = data_dict['Ks']
        self.indices = indices
        self.ray_type = ray_type
        self.batch_size = batch_size
        self.load2gpu_on_the_fly = load2gpu_on_the_fly
        
        # 이미지 크기 정보
        self.H, self.W = self.HW[0]  # 첫 번째 이미지 크기 (모든 이미지가 같다고 가정)
        
        # 모든 ray를 미리 계산하여 인덱스 생성
        self.total_pixels_per_image = self.H * self.W
        self.total_rays = len(self.indices) * self.total_pixels_per_image
        self.total_batches = (self.total_rays + self.batch_size - 1) // self.batch_size
        
        print(f"Dataset: {len(self.indices)} images, {self.total_rays:,} total rays, {self.total_batches} batches per epoch")

    def __len__(self):
        return self.total_batches

    def __getitem__(self, batch_idx):
        """
        배치 인덱스에 해당하는 ray들을 반환
        한 epoch에 모든 ray가 정확히 한 번씩 학습되도록 보장
        
        Args:
            batch_idx: 배치 인덱스 (0 ~ total_batches-1)
            
        Returns:
            dict: {
                'rgb': [batch_size, 3],
                'rays': [batch_size, 4] for twoplane,
                'img_indices': [batch_size]
            }
        """
        # 배치의 시작과 끝 ray 인덱스 계산
        start_ray_idx = batch_idx * self.batch_size
        end_ray_idx = min(start_ray_idx + self.batch_size, self.total_rays)
        actual_batch_size = end_ray_idx - start_ray_idx
        
        rgb_batch = []
        rays_batch = []
        img_indices_batch = []
        
        current_ray_idx = start_ray_idx
        
        while current_ray_idx < end_ray_idx:
            # 현재 ray가 속한 이미지와 픽셀 인덱스 계산
            img_idx_in_dataset = current_ray_idx // self.total_pixels_per_image
            pixel_idx_in_image = current_ray_idx % self.total_pixels_per_image
            
            actual_img_idx = self.indices[img_idx_in_dataset]
            
            # 이미지와 관련 정보 로드
            if isinstance(self.images, list):
                image = self.images[actual_img_idx]  # [H, W, 3]
            else:
                image = self.images[actual_img_idx]  # [H, W, 3]
            
            pose = self.poses[actual_img_idx]  # [3, 4]
            H, W = self.HW[actual_img_idx]
            K = self.Ks[actual_img_idx]  # [3, 3]
            
            # Ray 생성 (실시간 계산)
            if self.ray_type == 'twoplane':
                rays = get_rays_of_a_view_twoplane(H, W, K, pose)  # [H*W, 4]
            else:
                raise NotImplementedError(f'Ray type {self.ray_type} not implemented')
            
            # 이미지를 flatten
            rgb_flat = image.view(-1, 3)  # [H*W, 3]
            
            # 현재 이미지에서 처리할 픽셀 수 계산
            remaining_rays_in_batch = end_ray_idx - current_ray_idx
            remaining_pixels_in_image = self.total_pixels_per_image - pixel_idx_in_image
            pixels_to_process = min(remaining_rays_in_batch, remaining_pixels_in_image)
            
            # 해당 픽셀들 추출
            pixel_end_idx = pixel_idx_in_image + pixels_to_process
            rgb_subset = rgb_flat[pixel_idx_in_image:pixel_end_idx].cpu()  # [pixels_to_process, 3]
            rays_subset = rays[pixel_idx_in_image:pixel_end_idx].cpu()     # [pixels_to_process, 4]
            
            rgb_batch.append(rgb_subset)
            rays_batch.append(rays_subset)
            img_indices_batch.extend([actual_img_idx] * pixels_to_process)
            
            current_ray_idx += pixels_to_process
        
        # 배치 합치기
        rgb_batch = torch.cat(rgb_batch, dim=0)    # [actual_batch_size, 3]
        rays_batch = torch.cat(rays_batch, dim=0)  # [actual_batch_size, 4]
        
        return {
            'rgb': rgb_batch,
            'rays': rays_batch,
            'img_indices': img_indices_batch
        }


def create_lightfield_dataset(data_dict, split='train', batch_size=8192):
    """
    모든 ray를 한 epoch에 학습하는 Light Field Dataset 생성 함수
    
    Args:
        data_dict: load_data()에서 반환된 데이터 딕셔너리
        split: 'train', 'val', 'test' 중 하나
        batch_size: 배치당 ray 수
        
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
    
    # Dataset 생성 - 새로운 방식으로 모든 ray 포함
    dataset = LightFieldDataSet(
        data_dict=data_dict,
        indices=indices,
        ray_type='twoplane',  # 기본값
        batch_size=batch_size
    )
    
    # DataLoader 생성 - batch_size=1로 설정 (Dataset에서 이미 배치 처리됨)
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """
        이미 배치가 구성된 데이터를 그대로 반환
        """
        # batch는 길이 1인 리스트 (Dataset에서 이미 배치 구성됨)
        return batch[0]
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Dataset에서 이미 배치를 구성하므로 1
        shuffle=False,  # 순서대로 모든 ray 처리
        collate_fn=collate_fn,
        num_workers=0,  # 이미지가 이미 메모리에 있으므로
        pin_memory=False  # CUDA 호환성을 위해 False로 설정
    )
    
    return dataset, dataloader
