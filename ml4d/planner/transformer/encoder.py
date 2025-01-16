# encoder.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

from ml4d.planner.transformer.multi_head_attention import MultiHeadSelfAttention
from ml4d.planner.transformer.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """하나의 Transformer Encoder Layer"""
    hidden_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = False) -> jnp.ndarray:
        # 1. Multi-head self-attention
        attn_out = MultiHeadSelfAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )(x, mask=mask, deterministic=deterministic)
        # residual connection + layer norm
        x = x + nn.Dropout(rate=self.dropout_rate)(attn_out, deterministic=deterministic)
        x = nn.LayerNorm()(x)

        # 2. Feed Forward
        ff_out = FeedForward(
            hidden_dim=self.hidden_dim,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate
        )(x, deterministic=deterministic)
        x = x + nn.Dropout(rate=self.dropout_rate)(ff_out, deterministic=deterministic)
        x = nn.LayerNorm()(x)
        return x


class Encoder(nn.Module):
    """Transformer Encoder (N개의 EncoderLayer 쌓기)"""
    num_layers: int
    hidden_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, deterministic: bool = False) -> jnp.ndarray:
        for _ in range(self.num_layers):
            x = EncoderLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate
            )(x, mask=mask, deterministic=deterministic)
        return x
