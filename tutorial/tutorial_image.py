"""
Image Rectification Tutorial

이 튜토리얼은 카메라 캘리브레이션을 사용한 이미지 rectification을 다룹니다.
Rectification은 카메라 왜곡을 제거하고 이미지를 정규화된 뷰로 변환하는 과정입니다.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from pyquaternion import Quaternion


def quaternion_to_rotation_matrix(quaternion: list) -> np.ndarray:
    """Quaternion [w, x, y, z]을 rotation matrix로 변환.
    
    Args:
        quaternion: Quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    q = Quaternion(quaternion)
    return q.rotation_matrix


def get_camera_matrix_from_calib(calib: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """nuScenes 캘리브레이션 딕셔너리에서 카메라 행렬 추출.
    
    Args:
        calib: Camera calibration dictionary with keys:
            - intrinsic_matrix: 3x3 intrinsic matrix
            - translation: [x, y, z] translation vector
            - rotation: [w, x, y, z] quaternion
            
    Returns:
        Tuple of (K, R, t) where:
            K: 3x3 intrinsic matrix
            R: 3x3 rotation matrix
            t: 3x1 translation vector
    """
    K = np.array(calib["intrinsic_matrix"], dtype=np.float32)
    t = np.array(calib["translation"], dtype=np.float32).reshape(3, 1)
    R = quaternion_to_rotation_matrix(calib["rotation"])
    
    return K, R, t


def compute_rectification_maps(
    K: np.ndarray,
    D: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = (1600, 900),
    alpha: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Rectification 맵 계산.
    
    Args:
        K: 3x3 intrinsic matrix
        D: Distortion coefficients [k1, k2, p1, p2, k3] (optional)
        image_size: (width, height) of the image
        alpha: Free scaling parameter (0=no black borders, 1=all pixels valid)
        
    Returns:
        Tuple of (map1, map2) for cv2.remap()
    """
    width, height = image_size
    
    # Distortion coefficients가 없으면 0으로 설정
    if D is None:
        D = np.zeros(5, dtype=np.float32)
    
    # 새로운 카메라 행렬 계산 (alpha에 따라 조정)
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, D, (width, height), alpha, (width, height)
    )
    
    # Rectification 맵 계산
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, new_K, (width, height), cv2.CV_32FC1
    )
    
    return map1, map2, new_K


def rectify_image(
    image: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """이미지 rectification 적용.
    
    Args:
        image: Input image (BGR or RGB)
        map1: First map from compute_rectification_maps
        map2: Second map from compute_rectification_maps
        interpolation: Interpolation method (default: cv2.INTER_LINEAR)
        
    Returns:
        Rectified image
    """
    return cv2.remap(image, map1, map2, interpolation)


def stereo_rectify(
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    image_size: Tuple[int, int],
    R: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """스테레오 카메라 rectification.
    
    Args:
        K1, K2: Intrinsic matrices for camera 1 and 2
        D1, D2: Distortion coefficients for camera 1 and 2
        R1, R2: Rotation matrices for camera 1 and 2
        t1, t2: Translation vectors for camera 1 and 2
        image_size: (width, height) of the images
        R: Rotation matrix from camera 1 to camera 2 (optional)
        T: Translation vector from camera 1 to camera 2 (optional)
        
    Returns:
        Tuple of (R1_rect, R2_rect, P1, P2, Q, map1x, map1y, map2x, map2y)
    """
    width, height = image_size
    
    # R, T가 제공되지 않으면 두 카메라 포즈에서 계산
    if R is None or T is None:
        # Camera 1 to world
        R1_to_world = R1
        t1_to_world = t1
        
        # Camera 2 to world
        R2_to_world = R2
        t2_to_world = t2
        
        # World to camera 1
        R_world_to_1 = R1_to_world.T
        t_world_to_1 = -R1_to_world.T @ t1_to_world
        
        # Camera 2 to camera 1
        R = R_world_to_1 @ R2_to_world
        T = R_world_to_1 @ (t2_to_world - t1_to_world)
    
    # Stereo rectification
    R1_rect, R2_rect, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, (width, height), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.9
    )
    
    # Rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, D1, R1_rect, P1, (width, height), cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, D2, R2_rect, P2, (width, height), cv2.CV_32FC1
    )
    
    return R1_rect, R2_rect, P1, P2, Q, map1x, map1y, map2x, map2y


def example_single_camera_rectification():
    """단일 카메라 rectification 예제."""
    print("=" * 60)
    print("단일 카메라 Rectification 예제")
    print("=" * 60)
    
    # 예제 캘리브레이션 데이터 (nuScenes 형식)
    calib = {
        "intrinsic_matrix": [
            [1266.417203, 0.0, 816.267019],
            [0.0, 1266.417203, 491.507838],
            [0.0, 0.0, 1.0]
        ],
        "translation": [0.0, 0.0, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],  # [w, x, y, z]
        "original_image_size": [1600, 900]
    }
    
    # 카메라 행렬 추출
    K, R, t = get_camera_matrix_from_calib(calib)
    print(f"\nIntrinsic Matrix K:\n{K}")
    print(f"\nRotation Matrix R:\n{R}")
    print(f"\nTranslation t:\n{t}")
    
    # Distortion coefficients (예제 - 실제로는 캘리브레이션에서 가져와야 함)
    # nuScenes는 일반적으로 왜곡이 거의 없지만, 예제를 위해 작은 값을 사용
    D = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Rectification maps 계산
    map1, map2, new_K = compute_rectification_maps(
        K, D, calib["original_image_size"], alpha=0.0
    )
    print(f"\nNew Camera Matrix:\n{new_K}")
    
    # 예제 이미지 생성 (실제로는 nuScenes에서 로드)
    # 여기서는 체스보드 패턴을 생성하여 시각화
    image = np.ones((900, 1600, 3), dtype=np.uint8) * 255
    for i in range(0, 1600, 100):
        for j in range(0, 900, 100):
            if (i // 100 + j // 100) % 2 == 0:
                image[j:j+100, i:i+100] = [0, 0, 0]
    
    # Rectification 적용
    rectified = rectify_image(image, map1, map2)
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Rectified Image")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("rectification_example.png", dpi=150, bbox_inches='tight')
    print("\n결과 이미지가 'rectification_example.png'로 저장되었습니다.")
    plt.show()


def example_stereo_rectification():
    """스테레오 카메라 rectification 예제."""
    print("\n" + "=" * 60)
    print("스테레오 카메라 Rectification 예제")
    print("=" * 60)
    
    # 두 카메라의 캘리브레이션 (예: CAM_FRONT와 CAM_FRONT_LEFT)
    calib1 = {
        "intrinsic_matrix": [
            [1266.417203, 0.0, 816.267019],
            [0.0, 1266.417203, 491.507838],
            [0.0, 0.0, 1.0]
        ],
        "translation": [1.575, 0.0, 1.296],
        "rotation": [0.7071, 0.0, 0.0, 0.7071],  # 90도 회전 예제
    }
    
    calib2 = {
        "intrinsic_matrix": [
            [1266.417203, 0.0, 816.267019],
            [0.0, 1266.417203, 491.507838],
            [0.0, 0.0, 1.0]
        ],
        "translation": [1.575, 0.5, 1.296],  # 약간 오프셋
        "rotation": [0.7071, 0.0, 0.0, 0.7071],
    }
    
    K1, R1, t1 = get_camera_matrix_from_calib(calib1)
    K2, R2, t2 = get_camera_matrix_from_calib(calib2)
    
    D1 = np.zeros(5, dtype=np.float32)
    D2 = np.zeros(5, dtype=np.float32)
    
    image_size = (1600, 900)
    
    # Stereo rectification
    R1_rect, R2_rect, P1, P2, Q, map1x, map1y, map2x, map2y = stereo_rectify(
        K1, D1, K2, D2, R1, t1, R2, t2, image_size
    )
    
    print(f"\nRectified Rotation Matrix 1:\n{R1_rect}")
    print(f"\nRectified Rotation Matrix 2:\n{R2_rect}")
    print(f"\nProjection Matrix 1:\n{P1}")
    print(f"\nProjection Matrix 2:\n{P2}")
    print(f"\nDisparity-to-Depth Matrix Q:\n{Q}")
    
    print("\n스테레오 rectification 맵이 계산되었습니다.")
    print("이 맵들을 사용하여 두 카메라 이미지를 rectify할 수 있습니다.")


def example_with_nuscenes_calib(calib: Dict, image: Optional[np.ndarray] = None):
    """nuScenes 캘리브레이션을 사용한 실제 rectification 예제.
    
    Args:
        calib: nuScenes camera calibration dictionary
        image: Optional image array (BGR format). If None, a test pattern is generated.
    """
    print("\n" + "=" * 60)
    print("nuScenes 캘리브레이션을 사용한 Rectification")
    print("=" * 60)
    
    # 카메라 행렬 추출
    K, R, t = get_camera_matrix_from_calib(calib)
    
    # Distortion coefficients (nuScenes는 일반적으로 왜곡이 거의 없음)
    D = np.zeros(5, dtype=np.float32)
    
    # 이미지가 제공되지 않으면 테스트 패턴 생성
    if image is None:
        width, height = calib["original_image_size"]
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        # 그리드 패턴 생성
        for i in range(0, width, 50):
            cv2.line(image, (i, 0), (i, height), (200, 200, 200), 1)
        for j in range(0, height, 50):
            cv2.line(image, (0, j), (width, j), (200, 200, 200), 1)
    
    # Rectification maps 계산
    map1, map2, new_K = compute_rectification_maps(
        K, D, calib["original_image_size"], alpha=0.0
    )
    
    # Rectification 적용
    rectified = rectify_image(image, map1, map2)
    
    # 결과 반환
    return {
        "original": image,
        "rectified": rectified,
        "K_original": K,
        "K_rectified": new_K,
        "maps": (map1, map2)
    }


def example_regnet_backbone():
    """RegNet backbone 사용 예제."""
    print("\n" + "=" * 60)
    print("RegNet Backbone 사용 예제")
    print("=" * 60)
    
    try:
        import torch
        from src.models.backbones.backbone import RegNetBackbone, create_backbone
        
        # 방법 1: 직접 생성
        print("\n방법 1: RegNetBackbone 직접 생성")
        backbone1 = RegNetBackbone(
            model_name="regnetx_004",
            pretrained=True,
            features_only=True
        )
        print(f"Model: {backbone1.model_name}")
        print(f"Output channels: {backbone1.out_ch}")
        
        # 방법 2: Factory 함수 사용
        print("\n방법 2: create_backbone() 사용")
        backbone2 = create_backbone(
            backbone_type="regnet",
            model_name="regnety_008",
            pretrained=True
        )
        print(f"Model: {backbone2.model_name}")
        print(f"Output channels: {backbone2.out_ch}")
        
        # 예제 입력
        dummy_input = torch.randn(2, 3, 224, 224)
        print(f"\nInput shape: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = backbone1(dummy_input)
        print(f"Output shape: {output.shape}")
        
        print("\n사용 가능한 RegNet 모델:")
        print("  - regnetx_002, regnetx_004, regnetx_006, regnetx_008, regnetx_016")
        print("  - regnety_002, regnety_004, regnety_006, regnety_008, regnety_016")
        print("  - 더 많은 모델은 timm 라이브러리 문서 참조")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("timm 라이브러리가 필요합니다: pip install timm")


def example_bifpn_backbone():
    """BiFPN backbone 사용 예제."""
    print("\n" + "=" * 60)
    print("BiFPN Backbone 사용 예제")
    print("=" * 60)
    
    try:
        import torch
        from src.models.backbones.backbone import EfficientDetBackbone, BiFPN, create_backbone
        
        # 방법 1: EfficientDet (BiFPN 포함) 사용
        print("\n방법 1: EfficientDet (BiFPN 포함) 사용")
        efficientdet = EfficientDetBackbone(
            model_name="efficientdet_d0",
            pretrained=True,
            features_only=True
        )
        print(f"Model: {efficientdet.model_name}")
        print(f"Output channels: {efficientdet.out_ch}")
        
        # 예제 입력
        dummy_input = torch.randn(2, 3, 512, 512)
        print(f"\nInput shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = efficientdet(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # 방법 2: Standalone BiFPN 사용
        print("\n방법 2: Standalone BiFPN 사용")
        bifpn = BiFPN(
            num_channels=64,
            num_levels=5,
            num_bifpn_layers=3
        )
        
        # BiFPN은 여러 스케일의 feature map을 입력으로 받음
        # 예: P3, P4, P5, P6, P7
        dummy_features = [
            torch.randn(2, 64, 64, 64),   # P3
            torch.randn(2, 64, 32, 32),   # P4
            torch.randn(2, 64, 16, 16),   # P5
            torch.randn(2, 64, 8, 8),     # P6
            torch.randn(2, 64, 4, 4),     # P7
        ]
        
        print(f"Input features: {[f.shape for f in dummy_features]}")
        
        with torch.no_grad():
            refined_features = bifpn(dummy_features)
        
        print(f"Output features: {[f.shape for f in refined_features]}")
        
        # 방법 3: Factory 함수 사용
        print("\n방법 3: create_backbone() 사용")
        backbone3 = create_backbone(
            backbone_type="efficientdet",
            model_name="efficientdet_d1",
            pretrained=True
        )
        print(f"Model: {backbone3.model_name}")
        
        print("\n사용 가능한 EfficientDet 모델:")
        print("  - efficientdet_d0, efficientdet_d1, efficientdet_d2")
        print("  - efficientdet_d3, efficientdet_d4, efficientdet_d5")
        print("  - 더 많은 모델은 timm 라이브러리 문서 참조")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("timm 라이브러리가 필요합니다: pip install timm")


if __name__ == "__main__":
    # 단일 카메라 rectification 예제 실행
    example_single_camera_rectification()
    
    # 스테레오 rectification 예제 실행
    example_stereo_rectification()
    
    # RegNet backbone 예제 실행
    example_regnet_backbone()
    
    # BiFPN backbone 예제 실행
    example_bifpn_backbone()
    
    print("\n" + "=" * 60)
    print("Tutorial 완료!")
    print("=" * 60)
    print("\n사용법:")
    print("1. example_single_camera_rectification(): 단일 카메라 rectification")
    print("2. example_stereo_rectification(): 스테레오 카메라 rectification")
    print("3. example_with_nuscenes_calib(calib, image): nuScenes 데이터 사용")
    print("4. example_regnet_backbone(): RegNet pre-trained 모델 사용")
    print("5. example_bifpn_backbone(): BiFPN/EfficientDet pre-trained 모델 사용")
    print("\n함수들:")
    print("- get_camera_matrix_from_calib(calib): 캘리브레이션에서 행렬 추출")
    print("- compute_rectification_maps(K, D, image_size): rectification 맵 계산")
    print("- rectify_image(image, map1, map2): 이미지에 rectification 적용")
    print("- stereo_rectify(...): 스테레오 rectification")
    print("\nBackbone 사용:")
    print("- from src.models.backbones.backbone import create_backbone")
    print("- backbone = create_backbone('regnet', 'regnetx_004', pretrained=True)")
    print("- backbone = create_backbone('efficientdet', 'efficientdet_d0', pretrained=True)")

