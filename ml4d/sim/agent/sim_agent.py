import jax
import jax.numpy as jnp

from ml4d.sim.agent.agent import Agent
from ml4d.sim.agent.policy import (
    find_nearest_lane, find_front_vehicle)
from ml4d.sim.agent.idm import idm
from ml4d.sim.agent.pure_pursuit import pure_pursuit


class SimAgent(Agent):
    def __init__(self, rng: jax.Array, state: jax.Array, input: jax.Array):
        """
        Initializes agent state.

        Args:
            rng (jax.Array): A random key.
            state (jax.Array): The initial state (x, y, yaw, speed)
            input (jax.Array): The initial input (delta, accel)
        """
        super().__init__(rng, state, input)

    
    def action(self, rng: jax.Array) -> jax.Array:
        """
        Selects an action given the current simulator state.

        Args:
            rng (jax.Array): A random key.

        Returns:
            jax.Array: control input (delta, accel)
        """