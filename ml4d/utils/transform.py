import jax
import jax.numpy as jnp


def transform(agents: jax.Array, roadgraph: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Transforms agents and roadgraph coordinates to the frame of the first agent.
    
    Args:
        agents (jax.Array): Shape (batch, num_objects, state_dim).
            Each agent state includes (x, y, cos_h, sin_h, v, length, width).
        roadgraph (jax.Array): Shape (batch, num_lanes, num_points, 2).
            Each point is represented by (x, y).

    Returns:
        Tuple[jax.Array, jax.Array]: Transformed agents and roadgraph.
    """
    # Extract reference agent
    ref_agent = agents[:, 0, :]
    ref_x = ref_agent[:, 0]  # Shape: (batch,)
    ref_y = ref_agent[:, 1]  # Shape: (batch,)
    ref_cos_h = ref_agent[:, 2]
    ref_sin_h = ref_agent[:, 3]

    # Compute rotation matrix
    rot_matrix = jnp.stack(
        [
            jnp.stack([ref_cos_h, ref_sin_h], axis=-1),  # Shape: (batch, 2)
            jnp.stack([-ref_sin_h, ref_cos_h], axis=-1),  # Shape: (batch, 2)
        ],
        axis=-2,  # Shape: (batch, 2, 2)
    )

    # Rotation for roadgraph
    roadgraph_rotated = jnp.einsum('bij,bklj->bikl', rot_matrix, roadgraph)

    # Correct broadcasting issue by explicitly aligning shapes
    translation = jnp.stack([ref_x, ref_y], axis=-1)  # Shape: (batch, 2)
    translation = jnp.expand_dims(translation, axis=(1, 2))  # Shape: (batch, 1, 1, 2)

    # Translation after rotation
    roadgraph_transformed = roadgraph_rotated - translation

    # Rotation for agents
    agents_rotated = jnp.einsum('bij,bkj->bki', rot_matrix, agents[..., :2])

    # Translation after rotation
    agents_transformed_positions = agents_rotated - jnp.expand_dims(jnp.stack([ref_x, ref_y], axis=-1), axis=1)

    # Transform headings (cos_h, sin_h)
    headings = agents[..., 2:4]
    transformed_headings = jnp.einsum('bij,bkj->bki', rot_matrix, headings)

    # Combine transformed positions and headings
    agents_transformed = agents.at[..., :2].set(agents_transformed_positions)
    agents_transformed = agents_transformed.at[..., 2:4].set(transformed_headings)

    return agents_transformed, roadgraph_transformed