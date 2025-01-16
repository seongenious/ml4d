import jax
import jax.numpy as jnp

from ml4d.utils.unit import mod2pi


_MIN_LOOKAHEAD = 10.0

@jax.jit
def pure_pursuit(state: jax.Array, 
                 centerline: jax.Array, 
                 lookahead_time: float = 2.0,
                 wheelbase: float = 2.5) -> float:
    """Pure pursuit controller

    Args:
        state (jax.Array): (x, y, cos_h, sin_h, speed)
        centerline (jax.Array): (num_points, 2)
        lookahead_time (float): lookahead distance
        wheel_base (float): vehicle wheel base. Default is 2.845.

    Returns:
        jnp.float_: steering angle
    """
    # Get state and extend dimension
    x, y, v = state[..., 0], state[..., 1], state[..., 4]
    yaw = jnp.arctan2(state[..., 3], state[..., 2])
    
    # Compute feedback steering angle
    distances = jnp.sqrt(
      (centerline[..., 0] - x[..., None]) ** 2 + (centerline[..., 1] - y[..., None]) ** 2)
    lookahead_distance = _MIN_LOOKAHEAD + v * lookahead_time
    target_idx = jnp.argmin(
      jnp.abs(distances - lookahead_distance[..., None]), axis=-1)
    
    agent_idx = jnp.arange(state.shape[0])
    tx, ty = centerline[agent_idx, target_idx, 0], centerline[agent_idx, target_idx, 1]
    alpha = mod2pi(jnp.arctan2(ty - y, tx - x) - yaw)
    delta = jnp.arctan2(
      2.0 * wheelbase * jnp.sin(alpha), lookahead_distance)
    
    return delta