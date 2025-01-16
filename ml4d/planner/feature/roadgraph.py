import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any

class RoadGraphEncoder(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: shape (batch, 3, 100, 2)
        Returns:
            encoded_feat: shape (batch, hidden_dim)
        """

        # (batch, 3, 100, 2)를 CNN이 처리하기 좋게 재배치
        # 일반적인 conv2d는 (batch, channel, height, width) 형태가 필요
        # 여기서는 2를 channel로 보고, 3 x 100을 (height, width)로 볼 수도 있음
        # 예: (batch, 2, 3, 100) 형태로 reshape
        x = jnp.transpose(x, (0, 3, 1, 2))   # (batch, 2, 3, 100)

        # 예시: 간단한 CNN 블록 (커널, 스트라이드, 채널은 임의)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1,1))(x)  # (batch, 16, 1, 98)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(1, 3), strides=(1,1))(x)  # (batch, 32, 1, 96)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten -> (batch, ?)

        # 마지막에 Dense로 hidden_dim까지 축소
        x = nn.Dense(self.hidden_dim)(x)  # (batch, hidden_dim)
        return x
