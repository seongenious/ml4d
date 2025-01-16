#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints

from ml4d.planner.transformer.transformer import Transformer

# 1) 모델 초기화
def create_model(rng, vocab_size=32000, max_len=512, hidden_dim=512, num_heads=8, ff_dim=2048):
    model = Transformer(
        num_encoder_layers=6,
        num_decoder_layers=6,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        vocab_size=vocab_size,
        max_len=max_len,
        dropout_rate=0.1
    )
    # 임의의 입력으로 파라미터 초기화
    batch_size = 2
    enc_seq_len = 16
    dec_seq_len = 16
    dummy_encoder_input = jnp.ones((batch_size, enc_seq_len), dtype=jnp.int32)
    dummy_decoder_input = jnp.ones((batch_size, dec_seq_len), dtype=jnp.int32)
    
    params = model.init(
        rng,
        dummy_encoder_input,
        dummy_decoder_input,
        deterministic=True
    )['params']
    return model, params

# 2) loss function
def cross_entropy_loss(logits, labels):
    """단순 cross-entropy 예시."""
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    return loss

# 3) train step
@jax.jit
def train_step(state, rng, enc_in, dec_in, labels):
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            enc_in,
            dec_in,
            rngs={'dropout': rng},
            deterministic=False  # 훈련 중 dropout on
        )
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, logits

# 메인
def main():
    rng = jax.random.PRNGKey(42)

    # 모델/파라미터 생성
    model, params = create_model(rng)
    
    # Optax optimizer
    learning_rate = 1e-4
    tx = optax.adam(learning_rate)
    
    # TrainState
    state = train_state.TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params)
    )
    
    # 더미 배치
    enc_in = jnp.array([[1,2,3,4,0,0,0],[1,2,2,4,5,6,0]], dtype=jnp.int32)
    dec_in = jnp.array([[1,2,3,0,0],[1,2,2,3,4]], dtype=jnp.int32)
    labels = jnp.array([[2,3,4,5,0],[2,3,4,5,6]], dtype=jnp.int32)

    # 한 번의 train step
    rng, subkey = jax.random.split(rng)
    new_state, loss, logits = train_step(state, subkey, enc_in, dec_in, labels)
    print(f"Loss after one train step: {loss}")

if __name__ == "__main__":
    main()
