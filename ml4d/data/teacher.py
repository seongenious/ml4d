import jax
import jax.numpy as jnp
from jax import random

from ml4d.sim.agent.policy import find_nearest_lane, find_front_vehicle
from ml4d.sim.agent.idm import cruise, follow
from ml4d.sim.agent.pure_pursuit import pure_pursuit
from ml4d.utils.unit import kph2mps, deg2rad


MAX_DELTA_RATE = deg2rad(20)  # [rad/s]
MAX_JERK = 4.0  # [m/s^3]

def generate_init_policy(key: jax.random.PRNGKey,
                         agents: jax.Array,
                         delta: tuple = (deg2rad(-10.), deg2rad(10.)), 
                         accel: tuple = (-2., 2.)) -> jax.Array:
    key_delta, key_accel = random.split(key, 2)
    
    batch_size, num_objects, _ = agents.shape
    
    deltas = random.uniform(
        key_delta, shape=(batch_size, num_objects,), minval=delta[0], maxval=delta[1])
    accels = random.uniform(
        key_accel, shape=(batch_size, num_objects,), minval=accel[0], maxval=accel[1])
    
    valid = agents[..., -1]
    
    deltas = deltas * valid
    accels = accels * valid
    
    policy = jnp.stack([deltas, accels], axis=-1)
        
    return policy
    
    

def generate_keeping_policy(roadgraph: jax.Array,
                            agents: jax.Array,
                            lane_indices: jax.Array,
                            front_indices: jax.Array,
                            lookahead_time: float = 2.0,
                            wheelbase: float = 2.5,
                            speed_limit: float = kph2mps(50.0)) -> jax.Array:
    """
    Generate control input for lane keeping.

    Args:
        roadgraph (jax.Array): Roadgraph 
            shape: (batch_size, num_lanes, num_points, 2)
        agents (jax.Array): Agent states 
            shape: (batch_size, num_objects, state_dim+1)
            The last column is a valid flag (1 or 0).

    Returns:
        jax.Array: (batch_size, num_objects, input_dim+1)
        In this case, (batch_size, num_objects, 3) is assumed.
        Order: [delta (steering), accel (acceleration), valid]
    """
    # Compute delta and accel for all agents    
    def compute_policy(roadgraph, agents, lane_indices, front_indices, speed_limit):
        # Steering angle (delta) computation
        delta = pure_pursuit(
            state=agents, 
            centerline=roadgraph[lane_indices],
            lookahead_time=lookahead_time,
            wheelbase=wheelbase,
        )

        # Acceleration (accel) computation
        target = agents[front_indices]
        s0 = jnp.sqrt(
            (target[..., 0] - agents[..., 0])**2 + (target[..., 1] - agents[..., 1])**2
        )
        accel = jnp.where(
            front_indices != -1,
            follow(
                v=agents[..., 4],
                v0=speed_limit,
                s=s0,
                dv=target[..., 4] - agents[..., 4],
                delta=4.0,
            ),
            cruise(
                v=agents[..., 4], 
                v0=speed_limit,
                delta=4.0,
            ),
        )

        return delta, accel

    # Apply compute_policy for all agents in a batch
    deltas, accels = jax.vmap(
        compute_policy, in_axes=(0, 0, 0, 0, None))(
            roadgraph, 
            agents,
            lane_indices,
            front_indices,
            speed_limit,
        )
    valid = agents[..., -1]
    
    deltas = deltas * valid
    accels = accels * valid
    
    policy = jnp.stack([deltas, accels], axis=-1)
    
    return policy