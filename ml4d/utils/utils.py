import jax
import jax.numpy as jnp


def get_batched_index(agents: jax.Array) -> jax.Array:
    """
    Returns batched ego agent index

    Args:
        agents (jax.Array): _description_

    Returns:
        jax.Array: _description_
    """
    agents_valid = agents[..., -1]

    # Get the minimum index for each batch where agents_valid equals 1
    num_objects = agents_valid.shape[1]

    # Set invalid entries to a large number and find the minimum index
    masked_indices = jnp.where(
        agents_valid == 1, jnp.arange(num_objects), num_objects)
    min_indices = jnp.min(masked_indices, axis=1)

    # Replace indices greater than or equal to num_objects with -1 (no valid agent)
    min_indices = jnp.where(min_indices < num_objects, min_indices, -1)
    return min_indices.reshape(-1, 1)