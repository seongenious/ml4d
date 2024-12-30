import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ml4d.utils.geometry import agent2bbox
from ml4d.utils.unit import mps2kph


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
              policy: np.ndarray,
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
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot each lane of the roadgraph
    single_roadgraph = roadgraph[batch_index]  # (num_lanes, num_points, 2)
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
    # Returned shape: (num_objects, 4, 2)
    batch_agent = agents[batch_index:batch_index+1]
    agent_corners = agent2bbox(batch_agent)  # (1, num_objects, 4, 2)
    agent_corners = agent_corners[0]
    
    # Draw each agent as a polygon
    agent = batch_agent[0][:, -1]
    indices = np.where(agent == 1)[0]
    ego_index = np.where(indices.size > 0, indices[0], -1)

    for i, ac in enumerate(agent_corners):
        if agents[batch_index][i, -1] == 0: continue
        
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
        
        # center of corner points
        center_x = ac[:, 0].mean()
        center_y = ac[:, 1].mean()

        # Display speed
        speed = mps2kph(batch_agent[0, i, 4])  # speed = agents[..., 4]
        ax.text(center_x, center_y, f"{speed:.2f}", 
                color=edgecolor, 
                ha='center', va='center',
                fontsize=7)
    
    # Apply policy
    batch_policy = policy[batch_index:batch_index+1]
    batch_agent = simulate(batch_agent, batch_policy, dt=1.0)
    agent_corners = agent2bbox(batch_agent)  # (1, num_objects, 4, 2)
    agent_corners = agent_corners[0]
    
    # Draw each agent as a polygon
    agent = batch_agent[0][:, -1]
    indices = np.where(agent == 1)[0]
    ego_index = np.where(indices.size > 0, indices[0], -1)

    for i, ac in enumerate(agent_corners):
        if agents[batch_index][i, -1] == 0: continue
        
        edgecolor = GREEN if i == ego_index else RED
        edgecolor = GRAY
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
    ax.set_title(f"Roadgraph and Agents (Batch {batch_index})")
    ax.legend()
    ax.axis('equal')
    ax.grid(True)

    return fig


def simulate(agents: jax.Array, policy: jax.Array, dt: float = 0.1) -> jax.Array:
    WHEELBASE = 3.0
    
    c, s = agents[..., 2], agents[..., 3]
    theta = jnp.arctan2(s, c)

    agents = agents.at[..., 0].set(agents[..., 0] + agents[..., 4] * c * dt)
    agents = agents.at[..., 1].set(agents[..., 1] + agents[..., 4] * s * dt)
    
    theta = theta + agents[..., 4] / WHEELBASE * jnp.tan(policy[..., 0]) * dt
    agents = agents.at[..., 2].set(jnp.cos(theta))
    agents = agents.at[..., 3].set(jnp.sin(theta))
    
    agents = agents.at[..., 4].set(
        jnp.maximum(0., agents[..., 4] + policy[..., 1] * dt))
    
    return agents