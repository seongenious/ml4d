import jax
import jax.numpy as jnp
import numpy as np
from jax import random


def generate_centerline(start_point, curvature, num_points=100):
    """
    Generate a single center line with curvature.

    Args:
        start_point: jnp.ndarray - Start (x, y) coordinates of the center line
        curvature: float - Curvature applied to the center line
        num_points: int - Number of points in the center line

    Returns:
        jnp.ndarray - Center line points of shape (num_points, 2)
    """
    delta_x = 1.0  # Step size in the x-direction
    angles = jnp.linspace(0, curvature * (num_points - 1), num_points - 1)
    delta_y = jnp.sin(angles) * delta_x  # Curvature impact on y-direction

    # Compute cumulative sum of deltas to generate points
    deltas = jnp.column_stack((delta_x * jnp.cos(angles), delta_y))
    center_line = jnp.cumsum(deltas, axis=0)

    # Add the starting point
    center_line = jnp.vstack([start_point, center_line + start_point])
    return center_line


def generate_roadgraph(num_lanes: int,
                       start_points: jax.Array,
                       curvatures: jax.Array,
                       lane_spacing: float = 4.0,
                       num_points: int = 100):
    """
    Generate multiple batches of centerlines using jax.vmap.

    Args:
        num_lanes: int - Number of lanes per batch
        start_points: jnp.ndarray - Start points of shape
                                    (batch_size, 2)
        curvatures: jnp.ndarray - Curvature values of shape
                                  (batch_size, num_lanes)
        lane_spacing: float - Spacing between adjacent lanes
        num_points: int - Number of points in each center line

    Returns:
        jnp.ndarray - Generated centerlines of shape
                      (batch_size, num_lanes, num_points, 2)
    """
    # Compute start points for all lanes in a batch using lane_spacing
    def generate_start_points(start_point):
        y_offsets = jnp.arange(num_lanes) * lane_spacing
        lane_start_points = start_point + jnp.column_stack(
            (jnp.zeros(num_lanes), y_offsets))
        return lane_start_points  # Shape: (num_lanes, 2)

    # Partial function to generate a single lane
    def generate_lane(start_point, curvature):
        return generate_centerline(start_point, curvature, num_points)

    # Generate all lane start points for each batch
    batch_start_points = jax.vmap(generate_start_points)(start_points)

    # Map over num_lanes dimension (each lane within a batch)
    def generate_lanes_in_batch(start_points, curvature):
        return jax.vmap(
            generate_lane, in_axes=(0, None))(start_points, curvature)

    # Map over batch dimension
    roadgraph = jax.vmap(
        generate_lanes_in_batch)(batch_start_points, curvatures)
    return roadgraph  # Shape: (batch_size, num_lanes, num_points, 2)


def generate(file_path: str,
             batch_size: int = 8,
             num_lanes: int = 3,
             lane_spacing: float = 4.0,
             num_points: int = 100):
    """
    Generate and save center lines to a .npy file.

    Args:
        file_path: str - Path to save the .npy file
        batch_size: int - Number of batches to generate
        num_lanes: int - Number of lanes per batch
        num_points: int - Number of points in each center line
    """
    # Random generator setup
    key = random.PRNGKey(42)
    key_start, key_curvature = random.split(key)

    # Generate random start points for all batches and lanes
    start_points = random.uniform(
        key_start, shape=(batch_size, 2), minval=-5.0, maxval=5.0)

    # Generate random curvatures for all batches and lanes
    curvatures = random.uniform(
        key_curvature, shape=(batch_size,), minval=-0.01, maxval=0.01)

    # Generate roadgraph
    roadgraph = generate_roadgraph(
        num_lanes, start_points, curvatures, lane_spacing, num_points)

    # Save the generated data
    np.save(file_path, np.array(roadgraph))
    print(f"Center lines saved to {file_path} with shape {roadgraph.shape}")


if __name__ == "__main__":
    output_file = "./roadgraph.npy"  # Path to save the generated data
    generate(
        output_file,
        batch_size=10,
        num_lanes=5,
        lane_spacing=4.0,
        num_points=100
    )
