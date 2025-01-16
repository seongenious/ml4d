# feed_forward.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable


class FeedForward(nn.Module):
    """Positionwise feed-forward network."""
    hidden_dim: int
    ff_dim: int
    dropout_rate: float = 0.1
    activation_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """Applies a two-layer feed-forward transform."""
        # 첫 번째 dense
        x = nn.Dense(self.ff_dim)(x)
        x = self.activation_fn(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        # 두 번째 dense
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x
