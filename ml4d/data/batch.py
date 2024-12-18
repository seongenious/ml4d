import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from ml4d.data.agents import generate_agents
from ml4d.data.roadgraph import generate_roadgraph


def generate_batch(batch_size: int = 128) -> tuple[jax.Array, jax.Array]:
    """
    Generates a batch of roadgraphs and corresponding agents. 
    The roadgraphs are created using random start points and curvatures, 
    and agents are generated based on the resulting roadgraphs.

    Args:
        batch_size (int, optional): The number of roadgraphs and agent 
            sets to generate. Defaults to 128.

    Returns:
        Tuple[jax.Array, jax.Array]: 
        A tuple (roadgraph, agents), where:
          - roadgraph is a 4D jax.Array of shape 
            (batch_size, num_lanes, num_points, 2) 
            representing the generated roadgraphs.
          - agents is an array containing the generated agent data 
            associated with each roadgraph.
    """
    # Random generator setup
    key = random.PRNGKey(42)
    key_roadgraph, key_agents = random.split(key, 2)

    # Generate roadgraph
    roadgraph = generate_roadgraph(
        key=key_roadgraph, 
        batch_size=batch_size, 
        lane_spacing=4.0, 
        num_lanes=3, 
        num_points=100)
    
    # Generate agents
    agents = generate_agents(
        key=key_agents, 
        roadgraph=roadgraph, 
        num_objects=32)
    
    return roadgraph, agents


if __name__ == "__main__":
    roadgraph, agents = generate_batch(batch_size=128)
    
    # Save the generated data
    np.save("./roadgraph.npy", np.array(roadgraph))
    np.save("./agents.npy", np.array(agents))