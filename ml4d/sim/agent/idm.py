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


@jax.jit
def idm(v: jax.Array, 
        v0: jax.Array, 
        s: jax.Array, 
        dv: jax.Array, 
        delta: float = 4.0) -> jax.Array:
    """
    Compute command accelerations using IDM

    Args:
        v (jax.Array): Current speed
        v0 (jax.Array): Desired speed
        s (Optional[jax.Array], optional): distance to the front vehicle. 
            Defaults to None.
        dv (Optional[jax.Array], optional): speed difference to the front vehicle. 
            Defaults to None.
        delta (float, optional): acceleration decreasing rate. Defaults to 4.0.

    Returns:
        jax.Array: Acceleration commands
    """
    return jnp.where(
        s is None, cruise(v, v0, delta), follow(v, v0, s, dv, delta))