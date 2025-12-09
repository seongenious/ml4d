#!/usr/bin/env python3
"""
RT-1 모델 사용 예제
"""

import torch
import yaml
from src.models import RT1Model, create_rt1_model


def load_config(config_path: str = "src/conf/default.yaml"):
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dummy_data(batch_size: int = 2, num_cameras: int = 6, history_len: int = 4):
    """더미 데이터 생성"""
    # 멀티 카메라 이미지 시퀀스 [B, num_cameras, history_len, 3, H, W]
    images = torch.randn(batch_size, num_cameras, history_len, 3, 224, 224)
    
    # 프롬프트 토큰 [B, prompt_length]
    prompt_tokens = torch.randint(0, 1000, (batch_size, 20))
    
    return images, prompt_tokens


def main():
    """메인 함수"""
    print("RT-1 모델 사용 예제")
    print("=" * 50)
    
    # 설정 로드
    config = load_config()
    print(f"설정 로드 완료: {config['logging']['exp_name']}")
    
    # 모델 생성
    model = create_rt1_model(config)
    print(f"모델 생성 완료")
    print(f"- 카메라 수: {len(config['data']['cameras'])}")
    print(f"- 히스토리 길이: {config['data']['seq_len']}")
    print(f"- Transformer 레이어: {config['model']['transformer_layers']}")
    print(f"- 모델 차원: {config['model']['d_model']}")
    
    # 더미 데이터 생성
    images, prompt_tokens = create_dummy_data(
        batch_size=2,
        num_cameras=len(config['data']['cameras']),
        history_len=config['data']['seq_len']
    )
    print(f"\n입력 데이터 생성:")
    print(f"- 이미지 형태: {images.shape}")
    print(f"- 프롬프트 형태: {prompt_tokens.shape}")
    
    # 모델 추론
    model.eval()
    with torch.no_grad():
        actions = model(images, prompt_tokens)
        action_sequences = model.get_action_sequences(actions)
    
    print(f"\n모델 출력:")
    print(f"- 가속도 로짓: {actions['accel'].shape}")
    print(f"- 조향 로짓: {actions['steer'].shape}")
    print(f"- 가속도 확률: {action_sequences['accel_sequence'].shape}")
    print(f"- 조향 확률: {action_sequences['steer_sequence'].shape}")
    
    # 액션 시퀀스 해석
    print(f"\n액션 시퀀스 해석:")
    for i in range(2):  # 배치 크기만큼
        print(f"\n배치 {i+1}:")
        
        # 가속도 시퀀스 (가장 높은 확률의 클래스 선택)
        accel_pred = torch.argmax(action_sequences['accel_sequence'][i], dim=-1)
        print(f"  가속도 시퀀스: {accel_pred.tolist()}")
        
        # 조향 시퀀스 (가장 높은 확률의 클래스 선택)
        steer_pred = torch.argmax(action_sequences['steer_sequence'][i], dim=-1)
        print(f"  조향 시퀀스: {steer_pred.tolist()}")
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n모델 정보:")
    print(f"- 총 파라미터 수: {total_params:,}")
    print(f"- 학습 가능한 파라미터 수: {trainable_params:,}")
    
    # 메모리 사용량
    if torch.cuda.is_available():
        print(f"- GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    print("\n모델 테스트 완료!")


if __name__ == "__main__":
    main()
