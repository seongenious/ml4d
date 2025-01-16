# multi_head_attention.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer."""
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False
    ) -> jnp.ndarray:
        """Compute multi-head self-attention.
        
        Args:
            x: input of shape (batch_size, seq_len, hidden_dim).
            mask: optional attention mask of shape (batch_size, 1, seq_len, seq_len).
            deterministic: whether to apply dropout or not.
        """
        # Linen의 MultiHeadDotProductAttention 사용 (직접 구현도 가능)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=jnp.float32,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(x, x, mask=mask)
        return x


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention layer (decoder에서 사용)."""
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        context: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False
    ) -> jnp.ndarray:
        """Compute multi-head cross-attention between x and context.
        
        Args:
            x: decoder hidden states (batch_size, dec_seq_len, hidden_dim).
            context: encoder outputs (batch_size, enc_seq_len, hidden_dim).
            mask: optional attention mask (batch_size, 1, dec_seq_len, enc_seq_len).
            deterministic: whether to apply dropout or not.
        """
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=jnp.float32,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(x, context, mask=mask)
        return x
