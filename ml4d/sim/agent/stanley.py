import jax
import jax.numpy as jnp

from ml4d.utils.unit import kph2mps, mod2pi


_OFFSET_GAIN = 7.0
_MIN_SPEED = kph2mps(10)

@jax.jit
def stanley(state: jax.Array, 
            centerline: jax.Array) -> float:
    """Stanley controller

    Args:
        state (jax.Array): (x, y, cos_h, sin_h, speed)
        centerline (jax.Array): (num_points, 2)

    Returns:
        jnp.float_: steering angle
    """
    # Get state and extend dimension
    x, y, v = state[..., 0], state[..., 1], state[..., 4]
    yaw = mod2pi(jnp.arctan2(state[..., 3], state[..., 2]))
    
    # Compute feedback steering angle
    distances = jnp.sqrt(
      (centerline[..., 0] - x[..., None]) ** 2 + (centerline[..., 1] - y[..., None]) ** 2)
    target_idx = jnp.argmin(jnp.abs(distances), axis=-1)
    target_idx = jnp.clip(target_idx, a_min=0, a_max=centerline.shape[1] - 1)
    
    agent_idx = jnp.arange(state.shape[0])
    tx1, ty1 = centerline[agent_idx, target_idx, 0], centerline[agent_idx, target_idx, 1]
    tx2, ty2 = centerline[agent_idx, target_idx + 1, 0], centerline[agent_idx, target_idx + 1, 1]
    yaw_error = mod2pi(jnp.arctan2(ty2 - ty1, tx2 - tx1) - yaw)
    
    # Define 2D vectors
    vec = jnp.array([tx2 - tx1, ty2 - ty1])
    proj = jnp.array([x - tx1, y - ty1])

    # Calculate perpendicular distance
    num = vec[1] * proj[0] - vec[0] * proj[1]
    den = jnp.linalg.norm(v)
    offset_error = num / den
        
    delta = jnp.where(target_idx < centerline.shape[1] - 1, 
        yaw_error + jnp.arctan2(
            _OFFSET_GAIN * offset_error, jnp.maximum(v, _MIN_SPEED)),
        0.
    )   
    
    return delta