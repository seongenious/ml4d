import jax
import jax.numpy as jnp

from ml4d.utils.utils import get_batched_index


def rotation_matrix_2d(cos_h: jax.Array, sin_h: jax.Array) -> jax.Array:
    """
    Returns a 2D rotation matrix

    Args:
        cos_h (jax.Array): cosine theta array
        sin_h (jax.Array): sine theta array

    Returns:
        jax.Array: rotation matrix
    """
    return jnp.stack([cos_h, -sin_h, sin_h, cos_h], axis=-1).reshape(
        cos_h.shape + (2, 2))
    

def transform_points(matrix: jax.Array, pts: jax.Array) -> jax.Array:
    """
    Transforms points into new coordinates

    Args:
        matrix (jax.Array): Matrix representing the transformation 
            into the frame of pose of shape (prefix, dof+1, dof+1)
        pts (jax.Array): Points to translate of shape (prefix, ..., dof)

    Returns:
        jax.Array: Transfomred points
    """
    # Transform 2D points using a transformation matrix.
    dof = pts.shape[-1]
    pad_width = [(0, 1) if i == pts.ndim - 1 else (0, 0) for i in range(pts.ndim)]
    
    # Add homogeneous dimension
    out_pts = jnp.pad(pts, pad_width, mode='constant', constant_values=1.0)
    
    # Explicitly set high precision (for TPU)
    out_pts = out_pts.dot(matrix.T, precision='float32')[..., :dof]
    
    return out_pts


def transform_agents(agents: jax.Array) -> jax.Array:
    """
    Transform agents coordinate with respect to the ego agent

    Args:
        agents (jax.Array): agent states of shape 
            (batch size, num objects, state dim)

    Returns:
        jax.Array: Transformed agents
    """
    # Get ego agent index
    idx = get_batched_index(agents)    