import jax
import jax.numpy as jnp
from jax import random

from ml4d.utils.unit import deg2rad


def generate_centerline(start_point: jax.Array, 
                        heading_angle: jax.Array,
                        curvature: jax.Array,
                        num_points: int = 100) -> jax.Array:
    """
    Generates a single centerline given a start point and curvature. 
    The centerline consists of `num_points` evenly spaced points, 
    where each successive point is determined based on the given curvature.

    Args:
        start_point (jax.Array): A 1D array of shape (2,) 
            indicating the initial (x, y) position.
        heading angle (jax.Array): A 1D array of shape (1,)
            indicating the initial heading angle.
        curvature (jax.Array): A scalar that defines how the centerline curves. 
            Positive values curve left and negative values curve right.
        num_points (int, optional): The number of points to generate along 
            the centerline. Defaults to 100.

    Returns:
        jax.Array: A 2D array of shape (num_points, 2) representing the 
        (x, y) coordinates of the generated centerline.
    """
    step = 1.0  # Step size in the x-direction
    angles = heading_angle[..., None] + jnp.arange(num_points) * curvature * step

    # Compute cumulative sum of deltas to generate points
    deltas = jnp.column_stack((step * jnp.cos(angles), step * jnp.sin(angles)))
    centerline = jnp.cumsum(deltas, axis=0)

    # Add the starting point
    centerline = jnp.vstack([start_point, centerline + start_point])
    return centerline


def generate_roadgraph(key: jax.random.PRNGKey,
                       batch_size: int = 128,
                       num_lanes: int = 3,
                       lane_spacing: float = 4.0,
                       num_points: int = 100) -> jax.Array:
    """
    Generates a batch of roadgraphs, each containing multiple parallel lanes. 
    Each lane is represented by a centerline defined by a random start point 
    and curvature. The lanes within a single roadgraph are separated by 
    a fixed lane spacing.

    Args:
        key (jax.random.PRNGKey): A JAX random key for deterministic 
            random number generation.
        batch_size (int, optional): The number of roadgraphs to generate. 
            Defaults to 128.
        num_lanes (int, optional): The number of lanes per roadgraph. 
            Defaults to 3.
        lane_spacing (float, optional): The vertical spacing between adjacent lanes. 
            Defaults to 4.0.
        num_points (int, optional): The number of points defining each lane's centerline. 
            Defaults to 100.

    Returns:
        jax.Array: A 4D array of shape (batch_size, num_lanes, num_points, 2). 
        Each entry represents a batch of roadgraphs, containing the (x, y) 
        coordinates of each point along each lane's centerline.
    """
    key_start, key_heading, key_curvature = random.split(key, 3)
    
    # Generate random start points for all batches and lanes
    start_points = random.uniform(
        key_start, shape=(batch_size, 2), minval=-5.0, maxval=5.0)
    
    # Generate random heading angle for all batches and lanes
    heading_angles = random.uniform(
        key_heading, shape=(batch_size,), minval=deg2rad(-10), maxval=deg2rad(10))

    # Generate random curvatures for all batches and lanes
    curvatures = random.uniform(
        key_curvature, shape=(batch_size,), minval=-0.03, maxval=0.03)
    
    # Compute start points for all lanes in a batch using lane_spacing
    def generate_start_points(start_point: jax.Array):
        y_offsets = jnp.arange(num_lanes) * lane_spacing
        lane_start_points = start_point + jnp.column_stack(
            (jnp.zeros(num_lanes), y_offsets))
        return lane_start_points  # Shape: (num_lanes, 2)

    # Partial function to generate a single lane
    def generate_lane(start_point: jax.Array, 
                      heading_angles: jax.Array, 
                      curvature: jax.Array):
        return generate_centerline(start_point, 
                                   heading_angles, 
                                   curvature, 
                                   num_points)

    # Generate all lane start points for each batch
    batch_start_points = jax.vmap(generate_start_points)(start_points)

    # Map over num_lanes dimension (each lane within a batch)
    def generate_lanes_in_batch(start_points: jax.Array, 
                                heading_angles: jax.Array,
                                curvature: jax.Array):
        return jax.vmap(generate_lane, in_axes=(0, None, None))(
                start_points, heading_angles, curvature)

    # Map over batch dimension
    roadgraph = jax.vmap(generate_lanes_in_batch)(
            batch_start_points, heading_angles, curvatures)
    return roadgraph  # Shape: (batch_size, num_lanes, num_points, 2)