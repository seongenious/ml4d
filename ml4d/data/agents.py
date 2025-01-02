import jax
import jax.numpy as jnp
from jax import random

from ml4d.utils.unit import kph2mps, deg2rad, mod2pi
from ml4d.utils.geometry import compute_pairwise_overlaps


def generate_agents(key: jax.random.PRNGKey,
                    batch_size: int,
                    roadgraph: jax.Array,
                    num_objects: int = 32, 
                    noise: tuple = (2.0, 2.0, deg2rad(10.0)),
                    speed: tuple = (0, kph2mps(50)),
                    length: tuple = (4.8, 5.2),
                    width: tuple = (1.8, 2.2)) -> jax.Array:
    """
    Generates agent states for a given batch of roadgraphs. 
    Each agent is placed at a random point along the roadgraph, 
    with random heading, speed, steering angle (delta), acceleration, 
    and dimensions (length and width).

    Args:
        key (jax.random.PRNGKey): A random key for reproducible 
            random number generation.
        roadgraph (jax.Array): A 4D array of shape 
            (batch_size, num_lanes, num_points, 2) 
            representing the coordinates of the road segments.
        num_objects (int, optional): The number of agents to place per batch. 
            Defaults to 32.
        noise (tuple, optional): The range of random noise applied to the 
            agent pose (±noise). Defaults to (0.1, 0.1, deg2rad(5.0)).
        speed (tuple, optional): A (min, max) tuple for the agent speeds 
            in m/s. Defaults to (0, kph2mps(50)).
        length (tuple, optional): A (min, max) tuple for the agent length
            in meters. Defaults to (4.8, 5.2).
        width (tuple, optional): A (min, max) tuple for the agent width 
            in meters. Defaults to (1.8, 2.2).

    Returns:
        jax.Array: A 3D array of shape (batch_size, num_objects, state_dim), 
        where state_dim includes position (x, y), orientation (cos, sin), speed, 
        steering angle (delta), acceleration, length, width and validity
    """
    flattened_points = roadgraph.reshape(batch_size, -1, 2) 
    # Shape: (batch_size, num_lanes * num_points, 2)
    
    def generate_agents_per_batch(key: jax.random.PRNGKey,
                                  flattened_points: jax.Array) -> jax.Array:
        """
        Generate states for all agents in a single batch of roadgraph data.

        Args:
            key (jax.random.PRNGKey): A random key for generating random 
                properties.
            flattened_points (jax.Array): A 2D array of shape 
                (num_lanes * num_points, 2) 
                containing all the points along the road for a single batch instance.

        Returns:
            jax.Array: A 2D array of shape (num_objects, state_dim) 
            representing the generated agents.
        """
        key, key_pose, key_speed, key_size = random.split(key, 4)
        
        # Sample unique positions for agents
        num_points = flattened_points.shape[0]
        sampled_indices = random.choice(
            key_pose, num_points, shape=(num_objects,), replace=False)
        
        # Generate random position
        key_position, key_heading = random.split(key_pose)
        sampled_position = flattened_points[sampled_indices]  # Shape: (num_objects, 2)
        position_noise = random.uniform(
            key_position, shape=(num_objects, 2), minval=-jnp.array(noise[0:1]), maxval=jnp.array(noise[0:1]))
        sampled_position = sampled_position + position_noise
        
        # Generate random heading angle
        diff = jnp.diff(flattened_points, axis=0)
        heading = jnp.arctan2(diff[:, 1], diff[:, 0])
        heading = jnp.append(heading, heading[-1])  # Repeat last heading for same length
        
        sampled_heading = heading[sampled_indices]  # Sample headings for selected positions
        heading_noise = random.uniform(
            key_heading, shape=(num_objects,), minval=-noise[2], maxval=noise[2])
        sampled_heading = mod2pi(sampled_heading + heading_noise)
        cos_h = jnp.cos(sampled_heading)
        sin_h = jnp.sin(sampled_heading)
        
        # Generate random states for other attributes
        sampled_speed = random.uniform(
            key_speed, shape=(num_objects, 1), minval=speed[0], maxval=speed[1])
        sampled_length = random.uniform(
            key_size, shape=(num_objects, 1), minval=length[0], maxval=length[1])
        sampled_width = random.uniform(
            key_size, shape=(num_objects, 1), minval=width[0], maxval=width[1])
            
        # Combine all states into final state vector
        states = jnp.hstack([
            sampled_position, 
            cos_h[:, None], 
            sin_h[:, None], 
            sampled_speed,
            sampled_length, 
            sampled_width, 
        ])
        
        return states

    # Use vmap to parallelize agent state generation across batches
    keys = random.split(key, batch_size)
    agents = jax.vmap(generate_agents_per_batch)(keys, flattened_points)

    # Refine agents by checking overlap
    overlap = compute_pairwise_overlaps(agents=agents)
    invalid_mask = jnp.any(overlap, axis=-1)  # (batch_size, num_objects)
    
    # Convert True/False to 0/1, and invert so True->0, False->1
    valid = jnp.where(invalid_mask, 0.0, 1.0)  # (batch_size, num_objects)

    # Stack valid column to agents
    # agents shape: (batch_size, num_objects, state_dim)
    # valid shape: (batch_size, num_objects, 1)
    valid = valid[..., None]  # (batch_size, num_objects, 1)
    agents = agents * valid    
    agents = jnp.concatenate([agents, valid], axis=-1)
    
    return agents  # (batch_size, num_objects, state_dim+1)