import jax
import jax.numpy as jnp
from jax import random

from ml4d.sim.agent.policy import find_nearest_lane, find_front_vehicle
from ml4d.sim.agent.idm import cruise, follow
from ml4d.sim.agent.pure_pursuit import pure_pursuit
from ml4d.utils.unit import kph2mps, deg2rad


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
    
    policy = jnp.stack([deltas, accels, valid], axis=-1)
        
    return policy
    
    

def generate_keeping_policy(roadgraph: jax.Array,
                            agents: jax.Array,
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
    # Find the nearest lane index for each agent
    lane_indices = find_nearest_lane(roadgraph, agents)  # (batch, num_objects)

    # Find the front vehicle index for each agent
    front_indices = find_front_vehicle(agents, lane_indices)  # (batch, num_objects)

    # Combine lane_idx validity and agent_valid for the final validity check
    lane_valid = (lane_indices != -1).astype(jnp.float32)  # (batch, num_objects)
    agent_valid = agents[..., -1]  # (batch, num_objects)
    valid = lane_valid * agent_valid  # Both conditions must be valid

    # Compute delta and accel for all agents    
    def compute_policy(roadgraph, agents, lane_indices, front_indices, speed_limit):
        # Steering angle (delta) computation
        delta = pure_pursuit(agents, roadgraph[lane_indices])

        # Acceleration (accel) computation
        target = agents[front_indices]
        s0 = jnp.sqrt(
            (target[..., 0] - agents[..., 0])**2 + (target[..., 1] - agents[..., 1])**2
        )
        accel = jnp.where(
            front_indices != -1,
            follow(
                agents[..., 4],
                speed_limit,
                s0,
                target[..., 4] - agents[..., 4],
            ),
            cruise(agents[..., 4], speed_limit),
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
    
    policy = jnp.stack([deltas, accels, valid], axis=-1)
    
    return policy