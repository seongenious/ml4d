import abc
import jax
import jax.numpy as jnp


class Agent(abc.ABC):
    """
    Interface that defines agent functionality for inference.
    """
    
    WHEELBASE = 2.8
    
    def __init__(self, rng: jax.Array, state: jax.Array, input: jax.Array):
        """
        Initializes agent state.

        Args:
            rng (jax.Array): A random key.
            state (jax.Array): The initial state (x, y, yaw, speed)
            input (jax.Array): The initial input (delta, accel)
        """
        self.rng = rng
        self.state = state
        self.input = input
        

    @abc.abstractmethod
    def action(self, rng: jax.Array) -> jax.Array:
        """
        Selects an action given the current simulator state.

        Args:
            rng (jax.Array): A random key.

        Returns:
            jax.Array: control input (delta, accel)
        """


    def simulate(self, rng: jax.Array, dt: float):
        """
        Simulate agent one step using kinematic model.

        Args:
            rng (jax.Array): A random key.
            dt (float): time interval
        """
        self.input = self.action(rng)
        
        c, s = jnp.cos(self.state[..., 2]), jnp.sin(self.state[..., 2])
        
        self.state[..., 0] = self.state[..., 0] + self.state[..., 3] * c * dt
        self.state[..., 1] = self.state[..., 1] + self.state[..., 3] * s * dt
        self.state[..., 2] = self.state[..., 2] + \
            self.state[..., 3] / self.WHEELBASE * jnp.tan(self.input[..., 0]) * dt
        self.state[..., 3] = jnp.maximum(
            0., self.state[..., 3] + self.input[..., 1] * dt)