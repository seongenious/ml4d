# transformer.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

from ml4d.planner.transformer.encoder import Encoder
from ml4d.planner.transformer.decoder import Decoder


class Transformer(nn.Module):
    """Encoder-Decoder 구조를 모두 포함하는 Transformer"""
    num_encoder_layers: int
    num_decoder_layers: int
    hidden_dim: int
    num_heads: int
    ff_dim: int
    vocab_size: int
    max_len: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self,
                 encoder_input: jnp.ndarray,
                 decoder_input: jnp.ndarray,
                 encoder_mask: Optional[jnp.ndarray] = None,
                 decoder_self_mask: Optional[jnp.ndarray] = None,
                 decoder_cross_mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = False) -> jnp.ndarray:
        """
        Args:
            encoder_input: (batch_size, enc_seq_len)
            decoder_input: (batch_size, dec_seq_len)
            encoder_mask, decoder_self_mask, decoder_cross_mask: attention mask들.
        """

        # 1) 임베딩 + 위치 인코딩 (간단히 Dense로 대체 가능, 실제로는 embedding table 사용)
        emb_layer = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim)
        enc_emb = emb_layer(encoder_input)  # (batch_size, enc_seq_len, hidden_dim)
        dec_emb = emb_layer(decoder_input)  # (batch_size, dec_seq_len, hidden_dim)

        # (선택) 위치 인코딩 추가 가능. 여기서는 단순히 예시로 sinusoidal or learnable position embedding 생략
        # 예: enc_emb += PositionalEncoding(self.max_len, self.hidden_dim)(jnp.arange(enc_seq_len))
        #     dec_emb += PositionalEncoding(self.max_len, self.hidden_dim)(jnp.arange(dec_seq_len))

        # 2) Encoder
        enc_output = Encoder(
            num_layers=self.num_encoder_layers,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate
        )(enc_emb, mask=encoder_mask, deterministic=deterministic)

        # 3) Decoder
        dec_output = Decoder(
            num_layers=self.num_decoder_layers,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate
        )(
            dec_emb,
            enc_output,
            self_mask=decoder_self_mask,
            cross_mask=decoder_cross_mask,
            deterministic=deterministic
        )

        # 4) Output projection
        logits = nn.Dense(self.vocab_size)(dec_output)  # (batch_size, dec_seq_len, vocab_size)

        return logits
