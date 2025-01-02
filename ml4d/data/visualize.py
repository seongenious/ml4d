import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ml4d.utils.geometry import agent2bbox


RED = (0.9961, 0.4275, 0.451)
YELLOW = (1., 0.7961, 0.4667)
GREEN = (0.0902, 0.7647, 0.698)
BLUE = (0.1333, 0.4863, 0.6157)
WHITE = (1, 1, 1)
LIGHT_GRAY = (0.7, 0.7, 0.7)
GRAY = (0.5, 0.5, 0.5)
DARK_GRAY = (0.3, 0.3, 0.3)
BLACK = (0, 0, 0)

def visualize(roadgraph: np.ndarray, 
              agents: np.ndarray,
              batch_idx: int = 0):
    """
    Plots the roadgraph and agent positions for a given batch index.

    Args:
        roadgraph (np.ndarray): A 4D array of shape representing road 
            centerlines for each batch.(batch_size, num_lanes, num_points, 2) 
        agents (np.ndarray): A 3D array of shape containing agent states, 
            where the first two dimensions are (x, y) positions.
            (batch_size, num_objects, state_dim) 
        batch_idx (int, optional): Index of the batch to plot. Defaults to 0.

    Returns:
        None
    """    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot each lane of the roadgraph
    single_roadgraph = roadgraph[batch_idx]  # (num_lanes, num_points, 2)
    num_lanes = single_roadgraph.shape[0]
    for lane_idx in range(num_lanes):
        lane_points = single_roadgraph[lane_idx]
        x_coords = lane_points[:, 0]
        y_coords = lane_points[:, 1]
        ax.plot(x_coords, y_coords, 
                color=LIGHT_GRAY,
                linewidth=1.2,
                linestyle='--',
                label=f"Lane {lane_idx}")

    # Generate rectangle corners for each agent
    horizon = agents.shape[1]
    for h in range(horizon):
        temporal_agent = agents[batch_idx, h:h+1]
        agent_corners = agent2bbox(temporal_agent)  # (1, num_objects, 4, 2)
        agent_corners = agent_corners[0]

        # Draw each agent as a polygon
        agent = temporal_agent[0][:, -1]
        indices = np.where(agent == 1)[0]
        ego_index = np.where(indices.size > 0, indices[0], -1)

        for i, ac in enumerate(agent_corners):
            if agents[batch_idx, h, i, -1] == 0: continue
            
            edgecolor = GREEN if i == ego_index else RED
            linestyle = '-'
            
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
    ax.set_title(f"Roadgraph and Agents (Batch {batch_idx})")
    ax.legend()
    ax.axis('equal')
    ax.grid(True)

    return fig