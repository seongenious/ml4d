import jax.numpy as jnp


def kph2mps(speed: float):
    return speed / 3.6

def mps2kph(speed: float):
    return speed * 3.6

def rad2deg(angle: float):
    return angle * 180.0 / jnp.pi

def deg2rad(angle: float):
    return angle * jnp.pi / 180.0

def mod2pi(angle: float):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

def sec2ms(time: float):
    return time * 1000.0

def ms2sec(time: float):
    return time / 1000.0