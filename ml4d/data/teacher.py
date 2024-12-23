import jax
import jax.numpy as jnp

from ml4d.sim.agent.policy import (
    find_nearest_lane, find_front_vehicle)
from ml4d.sim.agent.idm import idm
from ml4d.sim.agent.pure_pursuit import pure_pursuit
from ml4d.utils.unit import kph2mps, deg2rad, mod2pi

def generate_keeping_policy(roadgraph: jax.Array,
                            agents: jax.Array) -> jax.Array:
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

    # Extract agent states
    agent_states = agents[..., :-1]  # (batch, num_objects, state_dim)

    # Compute delta and accel for all agents
    def compute_policy(batch_idx, agent_state, lane_idx, front_idx, valid_flag):
        # Steering angle (delta) computation
        delta = jnp.where(
            valid_flag, 
            pure_pursuit(agent_state, roadgraph[batch_idx, lane_idx]), 
            0.0
        )

        # Acceleration (accel) computation
        accel = jnp.where(
            valid_flag,
            jnp.where(
                front_indices != -1,
                idm(
                    agent_state[3],
                    kph2mps(50.0),
                    jnp.linalg.norm(agent_state[:2] - agents[front_idx, :2]),
                    agents[front_idx, 3] - agent_state[3],
                ),
                idm(agent_state[3], kph2mps(50.0), None, None),  # No front vehicle
            ),
            0.
        )

        return delta, accel

    # Apply compute_policy for all agents in a batch
    compute_policy_batched = jax.vmap(
        lambda batch_idx, agent_state, lane_idx, front_idx, valid_flag: compute_policy(
            batch_idx, agent_state, lane_idx, front_idx, valid_flag
        ),
        in_axes=(0, 0, 0, 0, 0),  # Process along batch dimension
    )

    deltas, accels = compute_policy_batched(
        jnp.arange(agents.shape[0]), agent_states, lane_indices, front_indices, valid)

    # Set invalid agents' outputs to 0
    deltas = deltas * valid
    accels = accels * valid

    # Final stacking
    deltas = deltas[..., None]  # (batch, num_objects, 1)
    accels = accels[..., None]  # (batch, num_objects, 1)
    valid = valid[..., None]   # (batch, num_objects, 1)

    policy = jnp.concatenate([deltas, accels, valid], axis=-1)  # (batch, num_objects, 3)
    return policy
