import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

class AgentsTransformer(nn.Module):
    hidden_dim: int
    num_heads: int
    num_layers: int
    max_seq_len: int = 1024   # 충분히 크게 잡아두기
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, agents: jnp.ndarray, deterministic: bool=False) -> jnp.ndarray:
        """
        Args:
            agents: shape (batch, T=30, N=32, state_dim=7)
        Returns:
            enc_out: shape (batch, T*N, hidden_dim) 
                     or (batch, hidden_dim) 로 풀링한 형태 (디자인에 따라 다름)
        """
        B, T, N, D = agents.shape

        # 1) Flatten time & agent 차원 -> (batch, T*N, D)
        x = agents.reshape((B, T*N, D))  # (batch, 960, 7)

        # 2) 임베딩(또는 간단한 Dense)으로 차원을 hidden_dim으로 매핑
        x = nn.Dense(self.hidden_dim)(x)  # (batch, 960, hidden_dim)

        # 3) (선택) position embedding
        # 만약 (시간 + 에이전트)라는 2차원 좌표를 어떻게 position으로 줄지 설계 필요
        # 여기서는 단순히 1D sinusoidal 등으로 seq_len=960에 대한 position encoding
        x = x + nn.Embed(num_embeddings=self.max_seq_len, features=self.hidden_dim)(
            jnp.arange(T*N)
        )

        # 4) 여러 층의 Transformer Encoder
        for _ in range(self.num_layers):
            # self-attention
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic
            )(x, x)
            x = x + nn.Dropout(rate=self.dropout_rate)(attn_out, deterministic=deterministic)
            x = nn.LayerNorm()(x)

            # feed-forward
            ff_out = nn.Dense(self.hidden_dim * 4)(x)
            ff_out = nn.relu(ff_out)
            ff_out = nn.Dropout(rate=self.dropout_rate)(ff_out, deterministic=deterministic)
            ff_out = nn.Dense(self.hidden_dim)(ff_out)
            x = x + nn.Dropout(rate=self.dropout_rate)(ff_out, deterministic=deterministic)
            x = nn.LayerNorm()(x)

        # 최종 출력: (batch, 960, hidden_dim)
        return x
