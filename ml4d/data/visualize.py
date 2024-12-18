import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ml4d.utils.geometry import agent2bbox, compute_pairwise_overlaps


def visualize(roadgraph: np.ndarray, 
              agents: np.ndarray, 
              batch_index: int = 0):
    """
    Plots the roadgraph and agent positions for a given batch index.

    Args:
        roadgraph (np.ndarray): A 4D array of shape representing road 
            centerlines for each batch.(batch_size, num_lanes, num_points, 2) 
        agents (np.ndarray): A 3D array of shape containing agent states, 
            where the first two dimensions are (x, y) positions.
            (batch_size, num_objects, state_dim) 
        batch_index (int, optional): Index of the batch to plot. Defaults to 0.

    Returns:
        None
    """    
    # Extract the specific batch
    single_roadgraph = roadgraph[batch_index]  # (num_lanes, num_points, 2)

    # Generate rectangle corners for each agent
    # Returned shape: (num_objects, 4, 2)
    agent_corners = agent2bbox(agents[batch_index:batch_index+1])  # (1, num_objects, 4, 2)
    agent_corners = agent_corners[0]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each lane of the roadgraph
    num_lanes = single_roadgraph.shape[0]
    for lane_idx in range(num_lanes):
        lane_points = single_roadgraph[lane_idx]
        x_coords = lane_points[:, 0]
        y_coords = lane_points[:, 1]
        ax.plot(x_coords, y_coords, label=f"Lane {lane_idx}")

    # Draw each agent as a polygon
    for i, ac in enumerate(agent_corners):
        edgecolor = 'black' if agents[batch_index][i, -1] > 0. else (0.7, 0.7, 0.7)
        linestyle = '-' if agents[batch_index][i, -1] > 0. else ':'
        
        # ac is (4, 2) array of corner points
        polygon = patches.Polygon(
            ac, 
            closed=True, 
            edgecolor=edgecolor, 
            facecolor='none', 
            linewidth=1.0, 
            linestyle=linestyle
        )
        ax.add_patch(polygon)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Roadgraph and Agents (Batch {batch_index})")
    ax.legend()
    ax.axis('equal')
    ax.grid(True)

    return fig


if __name__ == "__main__":
    # Set directory
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed")
    os.makedirs(dir, exist_ok=True)

    roadgraph = np.load(os.path.join(dir, "roadgraph.npy"))
    agents = np.load(os.path.join(dir, "agents.npy"))

    # Plot the first batch
    fig = visualize(roadgraph, agents, batch_index=10)

    # Save the figure to a file instead of showing it
    plt.savefig(os.path.join(dir, "visualize_batch.png"))
    plt.close(fig)
