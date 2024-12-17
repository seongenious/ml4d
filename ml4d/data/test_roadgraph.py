import numpy as np
import matplotlib.pyplot as plt
import random


def visualize_random_sample(input_file: str, output_file: str):
    """
    Visualize a random sample of center lines from a .npy file and save as PNG.

    Args:
        input_file: str - Path to the .npy file
        output_dir: str - Directory to save the visualization PNG file.
    """
    # Load data from the .npy file
    roadgraph = np.load(input_file)  # Shape: (batch, num_lanes, num_points, 2)
    batch_size, num_lanes, num_points, _ = roadgraph.shape
    print(f"Data loaded: shape {roadgraph.shape}")

    # Randomly select one batch from the data
    random_batch = random.randint(0, batch_size - 1)
    sample = roadgraph[random_batch]

    # Visualization
    plt.figure(figsize=(10, 6))
    for lane_idx, lane in enumerate(sample):
        plt.plot(lane[:, 0], lane[:, 1], label=f"Lane {lane_idx + 1}")

    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title(f"Visualization of Batch {random_batch}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # Save visualization as PNG
    plt.savefig(output_file)
    plt.close()
    print(f"Visualization saved as {output_file}")


if __name__ == "__main__":
    input_file = "./roadgraph.npy"  # Path to the generated .npy file
    output_file = "./roadgraph.png"
    visualize_random_sample(input_file, output_file)
