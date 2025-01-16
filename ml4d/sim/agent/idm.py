import jax
import jax.numpy as jnp


_MAX_ACCEL = 2.0
_COMFORT_DECEL = 4.0
_TIME_HEADWAY = 1.2
_MIN_GAP = 4.0

@jax.jit
def cruise(v: float, v0: float, delta: float = 4.0) -> float:
    """
    Compute cruise control command

    Args:
        v (float): Current speed
        v0 (float): Desired speed
        delta (float, optional): Decreasing rate. Defaults to 4.0.

    Returns:
        float: Command acceleration
    """
    return _MAX_ACCEL * (1 - (v / v0) ** delta)


@jax.jit 
def follow(
    v: float, v0: float, s: float, dv: float, delta: float = 4.0) -> float:
    """
    Compute following command 

    Args:
        v (float): Current speed
        v0 (float): Desired speed
        s (float): Distance to the front vehicle
        dv (float): Speed difference to the front vehicle
        delta (float): Acceleration decreasing rate

    Returns:
        jnp.float_: Command acceleration
    """
    s_star = _MIN_GAP + jnp.maximum(
        0.0, 
        v * _TIME_HEADWAY + v * dv / (2.0 * jnp.sqrt(_MAX_ACCEL * _COMFORT_DECEL)))
    
    return _MAX_ACCEL * (1.0 - (v / v0) ** delta - (s_star / s) ** 2.0)