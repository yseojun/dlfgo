import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import imageio
from lib import utils, dlfgo_model
from lib.dlfgo_model import DLFGO_twoplane
import time
import onnx
import onnxruntime as ort
# import onnxoptimizer


def to8b(x):
    """Convert [0,1] to [0,255]."""
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def config_parser():
    parser = argparse.ArgumentParser(description='Convert dlf2dgo model to ONNX')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output_path', type=str, default='model.onnx', help='Path to save the ONNX model')
    parser.add_argument('--test_image_path', type=str, default='test_image.png', help='Path to save the test image')
    parser.add_argument('--gpuid', type=int, default=1, help='GPU ID')
    return parser

def load_model(ckpt_path):
    print(f"Loading model from {ckpt_path}")
    
    # 원본 모델 로드
    original_model = utils.load_model(dlfgo_model.DLFGO_twoplane, ckpt_path)
    original_model.eval()
    
    
    return original_model

def get_test_rays(H, W):
    """
    테스트 이미지 생성을 위한 광선 생성 (twoplane 방식)
    """
    
    # 카메라 파라미터 설정 (빈 텐서)
    K = torch.zeros(3, 3, dtype=torch.float32)

    # 카메라 포즈 설정 (빈 텐서)
    c2w = torch.zeros(3, 4, dtype=torch.float32)
    
    # 카메라 위치 설정 (-1에서 1 사이의 값)
    c2w[0, 3] = 0.0 # x축 중심
    c2w[1, 3] = 0.0 # y축 중심
    # c2w[2, 3] = 0.0
    
    # 광선 생성
    ray = utils.get_rays_of_a_view_twoplane(H, W, K, c2w)

    return ray

def convert_to_onnx(model, output_path, H, W, device):
    # 테스트 입력 생성
    ray = get_test_rays(H, W)
    ray = ray.to(device)
    # 입력 형태 확인
    print(f"Input shape: {ray.shape}")
    
    # 모델 추론 테스트
    with torch.no_grad():
        output = model(ray)
    
    # 출력 형태 확인
    print(f"Output shape: {output.shape}")
    
    # 원본 모델로 이미지 생성
    rgb_numpy = output.reshape(H, W, 3).cpu().numpy()
    imageio.imwrite(f"torch_test.png", to8b(rgb_numpy))
    print(f"PyTorch test image saved to torch_test.png")
    
    # ONNX 변환
    torch.cuda.empty_cache()
    model.eval()
    torch.onnx.export(
        model,                                      # 모델
        ray,                                        # 모델 입력 (튜플 또는 텐서)
        output_path,                                # 저장 경로
        export_params=True,                         # 모델 파라미터 저장 여부
        opset_version=9 ,                           # ONNX 버전
        do_constant_folding=True,                   # 상수 폴딩 최적화 여부
        input_names=['ray'],                        # 입력 이름
        output_names=['rgb'],                       # 출력 이름
        verbose=False
    )
    
    print(f"ONNX model saved to {output_path}")
    # model = onnx.load(output_path)
    # optimized_model = onnxoptimizer.optimize(model)
    # onnx.save(optimized_model, 'model.onnx')
    
    return ray

def test_onnx_model(onnx_path, ray, H, W, test_image_path):
    """ONNX 모델 테스트"""
    
    # ONNX 런타임 세션 생성
    print(f"Testing ONNX model from {onnx_path}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # GPU 우선, CPU는 백업
    ort_session = ort.InferenceSession(onnx_path, providers=providers)

    # 입력 데이터 준비
    ort_inputs = {
        'ray': ray.cpu().numpy()  # ONNX Runtime은 numpy 입력을 기대
    }
    
    # 렌더링 시간 측정
    render_time0 = torch.cuda.Event(enable_timing=True)
    render_time1 = torch.cuda.Event(enable_timing=True)
    render_time0.record()
    
    # ONNX 모델 추론
    ort_outputs = ort_session.run(None, ort_inputs)
    
    render_time1.record()
    torch.cuda.synchronize()
    inference_time = torch.cuda.Event.elapsed_time(render_time0, render_time1)
    
    ort_output = ort_outputs[0]
    rgb_onnx = ort_output.reshape(H, W, 3)
    imageio.imwrite(test_image_path, to8b(rgb_onnx))
    print(f"ONNX test image saved to {test_image_path}")
    print(f"ONNX 모델 렌더링 시간: {inference_time}초")

def main():
    parser = config_parser()
    args = parser.parse_args()
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(args.gpuid)
        print('>>> Using GPU: {}'.format(args.gpuid))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # torch.set_default_tensor_type('torch.cuda.HalfTensor')
    else:
        print('>>> Using CPU')
        
    # 모델 로드
    model = load_model(args.ckpt_path)
    model = model.to(device)
    
    dataset_name = os.path.basename(args.ckpt_path).split('_')[0]
    decomp_name = os.path.basename(args.ckpt_path).split('_')[1]
    
    # 테스트 이미지 크기 설정
    if dataset_name == "beans":
        H = 512
        W = 256
    elif dataset_name == "bracelet":
        H = 512
        W = 256
    elif dataset_name == "gem":
        H = 384
        W = 512
    elif dataset_name == "truck":
        H = 640
        W = 480
    elif dataset_name == "chess":
        H = 700
        W = 400
    elif dataset_name == "bulldozer":
        H = 768
        W = 576
    elif dataset_name == "flowers":
        H = 640
        W = 768
    elif dataset_name == "treasure":
        H = 768
        W = 640
    else:
        H = 512
        W = 512
        
    # ONNX 변환
    ray = convert_to_onnx(
        model, 
        args.output_path, 
        H, 
        W,
        device
    )
    
    # ONNX 모델 테스트 및 이미지 생성
    try:
        test_onnx_model(
            args.output_path, 
            ray, 
            H, 
            W,
            args.test_image_path
        )
    except ImportError:
        print("onnxruntime not installed. Skipping ONNX model testing.")
        print("Install with: pip install onnxruntime")

if __name__ == "__main__":
    main() 