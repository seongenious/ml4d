import jax
import jax.numpy as jnp


def agent2bbox(agents: jax.Array) -> jax.Array:
    """
    Converts agent states to rectangular corner coordinates using JAX.
    
    Args:
        agents (jnp.ndarray): An array of shape (batch, num_objects, state_dim).
            We assume state = [x, y, cos_h, sin_h, speed, delta, accel, length, width].
            - x, y: Agent center position
            - cos_h, sin_h: Orientation of the agent
            - length, width: Dimensions of the agent
    
    Returns:
        jnp.ndarray: A (batch, num_objects, 4, 2) array of agent 
        corner coordinates. Each agent is represented by a rectangle 
        with four corners in global coordinates.
    """
    # Check shape
    agents = agents[..., :-1] if agents.shape[-1] == 10 else agents
    
    # Extract necessary fields
    x = agents[..., 0:1]          # (batch, num_objects, 1)
    y = agents[..., 1:2]          # (batch, num_objects, 1)
    cos_h = agents[..., 2:3]      # (batch, num_objects, 1)
    sin_h = agents[..., 3:4]      # (batch, num_objects, 1)
    l = agents[..., -2:-1]        # length  (batch, num_objects, 1)
    w = agents[..., -1:]           # width (batch, num_objects, 1)

    # Define local rectangle corners in a normalized coordinate system:
    # Here, (0,0) is the center of the vehicle.
    # The front of the vehicle is along the positive "length" direction.
    # The width direction is perpendicular to the length direction.
    local_corners = jnp.array([
        [-0.5, -0.5],  # Back-Left
        [-0.5,  0.5],  # Back-Right
         [0.5,  0.5],  # Front-Right
         [0.5, -0.5]   # Front-Left
    ], dtype=jnp.float32)  # (4, 2)

    # Expand dims to (1,1,4,2) so it can broadcast over batch and num_objects
    local_corners = local_corners[None, None, :, :]  # (1,1,4,2)

    # Scale factors for height and width: shape (batch, num_objects, 2)
    scale_factors = jnp.concatenate([l, w], axis=-1)  # (batch, num_objects, 2)

    # Expand scale_factors to (batch, num_objects, 1, 2)
    scale_factors = jnp.expand_dims(scale_factors, axis=2)  # (batch, num_objects, 1, 2)

    # Scale the local corners
    scaled_corners = local_corners * scale_factors  # (batch, num_objects, 4, 2)

    # Rotate
    # Rotation matrix:
    # [ cos_h -sin_h ]
    # [ sin_h  cos_h ]
    rotated_x = cos_h * scaled_corners[..., 0] - sin_h * scaled_corners[..., 1]
    rotated_y = sin_h * scaled_corners[..., 0] + cos_h * scaled_corners[..., 1]

    corners = jnp.stack([rotated_x, rotated_y], axis=-1)  # (batch, num_objects, 4, 2)

    # Translate by (x, y)
    corners = corners.at[..., 0].add(x)
    corners = corners.at[..., 1].add(y)

    return corners


def has_overlap(first: jnp.ndarray, second: jnp.ndarray) -> jnp.ndarray:
    """
    Checks overlap between two sets of agents (batch dimension only),
    first and second each have shape (state_dim, ), representing single agents.
    Returns a boolean array of shape (1,) indicating overlap.

    We use the Separating Axis Theorem (SAT) for OBB overlap check, 
    projecting corners onto axes defined by first agent's orientation.
    """
    # Extract cos_h, sin_h from first
    cos_h = first[..., 2]
    sin_h = first[..., 3]

    # Construct rotation normals_t from first agent orientation
    # normals_t: (batch, 2,2)
    normals_t = jnp.stack([
        jnp.stack([cos_h, -sin_h], axis=-1),
        jnp.stack([sin_h,  cos_h], axis=-1)
    ], axis=-2)

    # agent2bbox expects (batch, num_objects, state_dim), but we have (batch, state_dim)
    # Expand dims to (batch,1,state_dim)
    corners_a = agent2bbox(first[None, None, :])  # (batch,1,4,2)
    corners_b = agent2bbox(second[None, None, :]) # (batch,1,4,2)

    # Project A and B onto normals_t
    # proj: (...,4,2) x (...,2,2) -> (...,4,2)
    # Use jnp.matmul with broadcasting
    proj_a = jnp.matmul(corners_a, normals_t)  # (batch,1,4,2)
    proj_b = jnp.matmul(corners_b, normals_t)  # (batch,1,4,2)

    min_a = jnp.min(proj_a, axis=-2)  # (batch,1,2)
    max_a = jnp.max(proj_a, axis=-2)  # (batch,1,2)
    min_b = jnp.min(proj_b, axis=-2)  # (batch,1,2)
    max_b = jnp.max(proj_b, axis=-2)  # (batch,1,2)

    distance = jnp.minimum(max_a, max_b) - jnp.maximum(min_a, min_b)  # (batch,1,2)
    overlap_a = jnp.all(distance > 0, axis=-1).squeeze(axis=-1)  # (batch,)

    # Check overlap other way around (swap roles of first and second)
    cos_h2 = second[..., 2]
    sin_h2 = second[..., 3]
    normals_t2 = jnp.stack([
        jnp.stack([cos_h2, -sin_h2], axis=-1),
        jnp.stack([sin_h2,  cos_h2], axis=-1)
    ], axis=-2)

    proj_a2 = jnp.matmul(corners_a, normals_t2)  # (batch,1,4,2)
    proj_b2 = jnp.matmul(corners_b, normals_t2)  # (batch,1,4,2)

    min_a2 = jnp.min(proj_a2, axis=-2)
    max_a2 = jnp.max(proj_a2, axis=-2)
    min_b2 = jnp.min(proj_b2, axis=-2)
    max_b2 = jnp.max(proj_b2, axis=-2)

    distance2 = jnp.minimum(max_a2, max_b2) - jnp.maximum(min_a2, min_b2)
    overlap_b = jnp.all(distance2 > 0, axis=-1).squeeze(axis=-1)  # (batch,)

    return jnp.logical_and(overlap_a, overlap_b)


def compute_pairwise_overlaps(agents: jnp.ndarray) -> jnp.ndarray:
    """
    Computes pairwise overlap among all agents.
    agents: (batch, num_objects, state_dim)
    Returns: (batch, num_objects, num_objects) boolean array of overlaps.
    """        
    def unbatched_overlap(agents: jax.Array) -> jax.Array:
        check_overlap = jax.vmap(has_overlap, in_axes=(0, None))
        check_overlap = jax.vmap(check_overlap, in_axes=(None, 0))
        overlaps = check_overlap(agents, agents)
        overlaps = jnp.squeeze(overlaps, axis=-1)
        
        num_agents = agents.shape[0]
        self_mask = jnp.eye(num_agents)
        
        return jnp.where(self_mask, False, overlaps)

    return jax.vmap(unbatched_overlap, in_axes=0)(agents)