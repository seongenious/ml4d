import jax
import jax.numpy as jnp
from jax import random

from ml4d.utils.unit import deg2rad


def generate_centerlines(start_point: jnp.ndarray,
                         heading_angle: jnp.ndarray,
                         curvature: jnp.ndarray,
                         num_lanes: int = 3,
                         lane_spacing: float = 4.0,
                         num_points: int = 100) -> jnp.ndarray:
    """
    Generates multiple centerlines for lanes given a start point, heading angle, 
    and curvature. Each lane is spaced laterally by `lane_spacing`.

    Args:
        start_point (jnp.ndarray): A 1D array of shape (2,) 
            indicating the initial (x, y) position.
        heading_angle (jnp.ndarray): A scalar indicating the initial heading angle.
        curvature (jnp.ndarray): A scalar that defines how the centerline curves. 
            Positive values curve left and negative values curve right.
        num_lanes (int, optional): The number of lanes to generate. Defaults to 3.
        lane_spacing (float, optional): The lateral spacing between adjacent lanes. Defaults to 4.0.
        num_points (int, optional): The number of points to generate along 
            each centerline. Defaults to 100.

    Returns:
        jnp.ndarray: A 3D array of shape (num_lanes, num_points, 2) representing the 
        (x, y) coordinates of the generated centerlines for each lane.
    """
    # Step size in the x-direction
    step = 1.0

    # Compute angles for curvature over the points
    angles = heading_angle + jnp.arange(num_points) * curvature * step

    # Compute deltas for centerline generation
    deltas = jnp.column_stack((step * jnp.cos(angles), step * jnp.sin(angles)))

    # Generate centerline for the middle lane (reference lane)
    centerline = jnp.cumsum(deltas, axis=0)
    centerline = jnp.vstack([start_point, centerline + start_point])

    # Compute offsets for additional lanes
    offsets = jnp.arange(0, num_lanes) * lane_spacing

    # Unit vector perpendicular to the heading direction (lateral direction)
    lateral_direction = jnp.column_stack([-jnp.sin(angles), jnp.cos(angles)])

    # Apply offsets to generate all lanes
    offset_vectors = offsets[:, None, None] * lateral_direction[None, :, :]
    centerlines = centerline[None, :-1, :] + offset_vectors

    return centerlines


def generate_roadgraph(key: jax.random.PRNGKey,
                       batch_size: int = 128,
                       num_lanes: int = 3,
                       lane_spacing: float = 4.0,
                       num_points: int = 100,
                       position: tuple = (-0.0, 0.0),
                       heading: tuple = (deg2rad(-0), deg2rad(0)),
                       curvature: tuple = (-0.0, 0.0)) -> jax.Array:
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
        lane_spacing (float, optional): The vertical spacing between
            adjacent lanes. Defaults to 4.0.
        num_points (int, optional): The number of points defining each
            lane's centerline. Defaults to 100.

    Returns:
        jax.Array: A 4D array of shape (batch_size, num_lanes, num_points, 2).
        Each entry represents a batch of roadgraphs, containing the (x, y)
        coordinates of each point along each lane's centerline.
    """
    key_start, key_heading, key_curvature = random.split(key, 3)
    
    # Generate random start points for all batches and lanes
    start_points = random.uniform(
        key_start, shape=(batch_size, 2), minval=position[0], maxval=position[1])
    
    # Generate random heading angle for all batches and lanes
    heading_angles = random.uniform(
        key_heading, shape=(batch_size,), minval=heading[0], maxval=heading[1])

    # Generate random curvatures for all batches and lanes
    curvatures = random.uniform(
        key_curvature, shape=(batch_size,), minval=curvature[0], maxval=curvature[1])

    # Map over batch dimension, Shape: (batch_size, num_lanes, num_points, 2)
    return jax.vmap(
        generate_centerlines, in_axes=(0, 0, 0, None, None, None))(
            start_points,
            heading_angles,
            curvatures,
            num_lanes,
            lane_spacing,
            num_points
        )