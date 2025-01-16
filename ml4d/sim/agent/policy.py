import jax
import jax.numpy as jnp


def find_nearest_lane(roadgraph: jax.Array, agents: jax.Array) -> jax.Array:
    """
    Find nearest centerline lane index for each agent.

    Args:
        roadgraph (jax.Array): Roadgraph centerlines 
            with shape (batch_size, num_lanes, num_points, 2).
        agents (jax.Array): Agent states 
            with shape (batch_size, num_objects, state_dim+1).
            - The last column is a valid flag (1 for valid, 0 for invalid).

    Returns:
        jax.Array: Nearest lane index per agent.
        Shape: (batch_size, num_objects)
        For invalid agents, returns -1.
    """
    # Extract positions and valid flags
    # agents[..., :2]: x,y position of agents (batch_size, num_objects, 2)
    # agents[..., -1]: valid flag (batch_size, num_objects)
    agent_positions = agents[..., :2]

    # roadgraph: (batch_size, num_lanes, num_points, 2)
    # Compute distances:
    # We want to broadcast agent positions against all lane points.
    # agent_positions: (batch_size, num_objects, 2)
    # Expand dimensions to match for broadcasting:
    # We'll get a shape: (batch_size, num_objects, num_lanes, num_points)
    # (batch, obj, 1, 1, 2)
    agent_positions_expanded = agent_positions[:, :, None, None, :]
    # (batch, 1, num_lanes, num_points, 2)
    lane_points = roadgraph[:, None, :, :, :]
    
    # Compute squared distances and then sqrt
    # (batch, obj, num_lanes, num_points, 2)
    diff = agent_positions_expanded - lane_points
    # (batch, obj, num_lanes, num_points)
    dist_sq = jnp.sum(diff**2, axis=-1)
    dist = jnp.sqrt(dist_sq)

    # Find minimum distance lane for each agent
    # First, min over num_points to get min distance per lane
    min_dist_per_lane = jnp.min(dist, axis=-1)  # (batch, obj, num_lanes)
    # Then, argmin over lanes to find closest lane
    lane_idx = jnp.argmin(min_dist_per_lane, axis=-1)  # (batch, obj)

    return lane_idx


def find_front_vehicle(agents: jax.Array, lane_idx: jax.Array) -> jax.Array:
    """
    Find the index of the closest front vehicle for each agent. 
    The 'front' is defined in terms of the agent's heading direction.

    If no front vehicle is found, returns -1.

    Args:
        agents (jax.Array): (batch_size, num_objects, state_dim+1)
            Last column is valid flag (1 for valid agent, 0 for invalid).
        lane_idx (jax.Array): (batch_size, num_objects)
    Returns:
        jnp.ndarray: (batch_size, num_objects) with the index of the 
            front vehicle for each agent. -1 if no front vehicle is found.
    """
    # Extract necessary fields
    pos = agents[..., :2]     # (batch, obj, 2)
    cos_h = agents[..., 2]
    sin_h = agents[..., 3]
    valid = agents[..., -1]   # (batch, obj)

    # Heading vector for each agent
    heading = jnp.stack([cos_h, sin_h], axis=-1)  # (batch, obj, 2)

    num_objects = agents.shape[1]

    # Compute relative positions: pos_j - pos_i
    # pos: (batch, obj, 2)
    # Expand dims to compare every pair (i,j)
    # rel_pos[i,j] = pos_j - pos_i
    # (batch, obj_i, obj_j, 2)
    rel_pos = (pos[:, None, :, :] - pos[:, :, None, :])

    # Project rel_pos onto heading of agent i
    # heading_i: (batch, obj_i, 1, 2)
    heading_i = heading[:, :, None, :]
    # (batch, obj_i, obj_j)
    projection = jnp.sum(rel_pos * heading_i, axis=-1)

    # Conditions:
    # 1. valid[j] > 0:
    # 2. lane_idx[i] == lane_idx[j]:
    # 3. i != j:
    # 4. projection > 0:
    i_idx = jnp.arange(num_objects)
    j_idx = jnp.arange(num_objects)
    # Broadcast to shape (obj_i, obj_j)
    i_idx = i_idx[None, :]
    j_idx = j_idx[:, None]

    # (batch, obj_i, obj_j)
    same_lane = (lane_idx[:, :, None] == lane_idx[:, None, :])
    not_self = jnp.logical_not(jnp.eye(num_objects, dtype=bool)[None, :, :])
    valid_j = (valid[:, None, :] > 0)
    front = (projection > 0)

    mask = same_lane & not_self & valid_j & front
    proj_masked = jnp.where(mask, projection, jnp.inf)

    # nearest front vehicle index
    front_idx = jnp.argmin(proj_masked, axis=-1)  # (batch, obj_i)

    min_proj = jnp.min(proj_masked, axis=-1)
    front_idx = jnp.where(min_proj < jnp.inf, front_idx, -1)

    return front_idx
