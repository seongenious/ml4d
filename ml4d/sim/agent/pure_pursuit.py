import jax
import jax.numpy as jnp


_MIN_LOOKAHEAD = 5.0

@jax.jit
def pure_pursuit(state: jax.Array, 
                 centerline: jax.Array, 
                 lookahead_time: float = 2.0,
                 wheelbase: float = 3.0) -> float:
    """Pure pursuit controller

    Args:
        state (jax.Array): (x, y, cos_h, sin_h, speed)
        centerline (jax.Array): (num_points, 2)
        lookahead_time (float): lookahead distance
        wheel_base (float): vehicle wheel base. Default is 2.845.

    Returns:
        jnp.float_: steering angle
    """
    # Compute feedback steering angle
    print(state.shape)
    x, y, cos_h, sin_h, v = state[..., 0], state[..., 1], state[..., 2], state[..., 3], state[..., 4]
    yaw = jnp.arctan2(sin_h, cos_h)
    
    print(centerline.shape)
    distances = jnp.sqrt(
      (centerline[..., 0] - x) ** 2 + (centerline[..., 1] - y) ** 2)
    lookahead_distance = _MIN_LOOKAHEAD + v * lookahead_time
    target_idx = jnp.argmin(jnp.abs(distances - lookahead_distance))
    
    tx, ty = centerline[..., target_idx, 0], centerline[..., target_idx, 1]
    alpha = jnp.arctan2(ty - y, tx - x) - yaw
    delta = jnp.arctan2(
      2.0 * wheelbase * jnp.sin(alpha), lookahead_distance)
    
    return delta