# decoder.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

from ml4d.planner.transformer.multi_head_attention import (
    MultiHeadSelfAttention, MultiHeadCrossAttention)
from ml4d.planner.transformer.feed_forward import FeedForward


class DecoderLayer(nn.Module):
    """하나의 Transformer Decoder Layer"""
    hidden_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 enc_output: jnp.ndarray,
                 self_mask: Optional[jnp.ndarray] = None,
                 cross_mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = False) -> jnp.ndarray:
        # 1. Masked Multi-head self-attention (미래 토큰을 가리는 mask)
        self_attn_out = MultiHeadSelfAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )(x, mask=self_mask, deterministic=deterministic)
        x = x + nn.Dropout(rate=self.dropout_rate)(self_attn_out, deterministic=deterministic)
        x = nn.LayerNorm()(x)

        # 2. Multi-head cross-attention (encoder output과 decoder hidden을 인풋)
        cross_attn_out = MultiHeadCrossAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )(x, enc_output, mask=cross_mask, deterministic=deterministic)
        x = x + nn.Dropout(rate=self.dropout_rate)(cross_attn_out, deterministic=deterministic)
        x = nn.LayerNorm()(x)

        # 3. Feed Forward
        ff_out = FeedForward(
            hidden_dim=self.hidden_dim,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate
        )(x, deterministic=deterministic)
        x = x + nn.Dropout(rate=self.dropout_rate)(ff_out, deterministic=deterministic)
        x = nn.LayerNorm()(x)

        return x


class Decoder(nn.Module):
    """Transformer Decoder (N개의 DecoderLayer 쌓기)"""
    num_layers: int
    hidden_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 enc_output: jnp.ndarray,
                 self_mask: Optional[jnp.ndarray] = None,
                 cross_mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = False) -> jnp.ndarray:
        for _ in range(self.num_layers):
            x = DecoderLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate
            )(
                x, 
                enc_output,
                self_mask=self_mask,
                cross_mask=cross_mask,
                deterministic=deterministic
            )
        return x
