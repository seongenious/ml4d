import jax
import jax.numpy as jnp

from ml4d.utils.utils import get_batched_ego_index


def rotation_matrix_2d(cos_h: jax.Array, sin_h: jax.Array) -> jax.Array:
    """
    Returns a 2D rotation matrix

    Args:
        cos_h (jax.Array): cosine theta array
        sin_h (jax.Array): sine theta array

    Returns:
        jax.Array: rotation matrix
    """
    return jnp.stack([cos_h, sin_h, -sin_h, cos_h], axis=-1).reshape(
        cos_h.shape + (2, 2))


def transform_headings(cos_h: jax.Array, sin_h: jax.Array, headings: jax.Array) -> jax.Array:
    """
    Returns transformed headings according to cos_h, sin_h from ego agents

    Args:
        cos_h (jax.Array): Reference cosine theta of agents. Shape: (batch,)
        sin_h (jax.Array): Reference sine theta of agents. Shape: (batch,)
        headings (jax.Array): Heading of all agents. Shape: (batch, num objects, 2(=cos_h + sin_h))

    Returns:
        jax.Array: Transformed headings. Shape: (batch, num objects, 2)
    """
    # Compute rotation matrix from ego agent headings
    rotation_matrix = rotation_matrix_2d(cos_h, sin_h)  # Shape: (batch, 2, 2)

    # Rotate heading vectors
    transformed_headings = jnp.einsum("bij,bkj->bki", rotation_matrix, headings)  # Shape: (batch, num_objects, 2)

    return transformed_headings


def transform_agents(agents: jax.Array) -> jax.Array:
    """
    Transform agents coordinate with respect to the ego agent

    Args:
        agents (jax.Array): agent states of shape 
            (batch size, num objects, state dim)

    Returns:
        jax.Array: Transformed agents
    """
    # Get ego agent indices for each batch
    ego_indices = get_batched_ego_index(agents)  # Shape: (batch_size,)

    # Use advanced indexing to select ego agents
    ego_agent = agents[jnp.arange(agents.shape[0]), ego_indices]  # Shape: (batch_size, state_dim)

    # Extract ego position and orientation
    ego_x = ego_agent[:, 0]  # Shape: (batch_size,)
    ego_y = ego_agent[:, 1]  # Shape: (batch_size,)
    ego_cos_h = ego_agent[:, 2]  # Shape: (batch_size,)
    ego_sin_h = ego_agent[:, 3]  # Shape: (batch_size,)

    # Transform points    
    rotation_matrix = rotation_matrix_2d(ego_cos_h, ego_sin_h)
    translations = jnp.expand_dims(jnp.stack([ego_x, ego_y], axis=-1), axis=1)
    
    positions = agents[..., :2]  # Shape: (batch, num_objects, 2)
    translated_positions = positions - translations
    
    transformed_positions = rotation_matrix @ translated_positions.transpose(0, 2, 1)
    transformed_positions = transformed_positions.transpose(0, 2, 1)
    
    # Transform headings
    headings = agents[..., 2:4]  # Shape: (batch, num_objects, 2)
    transformed_headings = transform_headings(ego_cos_h, ego_sin_h, headings)  # Shape: (batch, num_objects, 2)
    
    # Combine transformed positions and headings with the rest of the state
    transformed_agents = agents.at[..., :2].set(transformed_positions)  # Update positions
    transformed_agents = transformed_agents.at[..., 2:4].set(transformed_headings)  # Update headings

    return transformed_agents


def transform_roadgraph(roadgraph: jax.Array, agents: jax.Array) -> jax.Array:
    """
    Transform roadgraph points with respect to the ego agent

    Args:
        roadgraph (jax.Array): roadgraph of shape
            (batch size, num lanes, num points, state dim)
        agents (jax.Array): agent states os shape
            (batch size, num objects, state dim)

    Returns:
        jax.Array: Transformed roadgraph. 
            Shape: (batch, num lanes, num points, state dim)
    """
    # Get ego agent indices for each batch
    ego_indices = get_batched_ego_index(agents)  # Shape: (batch_size,)

    # Use advanced indexing to select ego agents
    ego_agent = agents[jnp.arange(agents.shape[0]), ego_indices]  # Shape: (batch_size, state_dim)

    # Extract ego position and orientation
    ego_x = ego_agent[:, 0]  # Shape: (batch_size,)
    ego_y = ego_agent[:, 1]  # Shape: (batch_size,)
    ego_cos_h = ego_agent[:, 2]  # Shape: (batch_size,)
    ego_sin_h = ego_agent[:, 3]  # Shape: (batch_size,)
    
    # Get attributes
    batch_size, num_lanes, num_points, _ = roadgraph.shape

    # Transform points
    rotation_matrix = rotation_matrix_2d(ego_cos_h, ego_sin_h)
    translations = jnp.expand_dims(jnp.stack([ego_x, ego_y], axis=-1), axis=(1, 2))
    
    positions = roadgraph[..., :2]  # Shape: (batch, num lanes, num points, 2)
    translated_positions = positions - translations
    translated_positions = translated_positions.reshape(batch_size, -1, 2)    
    
    transformed_positions = rotation_matrix @ translated_positions.transpose(0, 2, 1)
    transformed_positions = transformed_positions.transpose(0, 2, 1)
    
    transformed_positions = transformed_positions.reshape(
        batch_size, num_lanes, num_points, 2)
    
    return transformed_positions