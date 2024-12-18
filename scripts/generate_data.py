#!/usr/bin/env python3

import os
import argparse
import time
from datetime import datetime
import logging

import jax
import numpy as np
from jax import random
import matplotlib.pyplot as plt

from ml4d.data.agents import generate_agents
from ml4d.data.roadgraph import generate_roadgraph
from ml4d.data.visualize import visualize
from ml4d.utils.transform import transform


# Suppress JAX logs below WARNING
logging.getLogger("jax").setLevel(logging.WARNING)

# Suppress matplotlib logs below WARNING
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Setup logging
logging.basicConfig(
    level=-logging.INFO,
    format="[%(levelname)s] [%(asctime)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

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
    logging.info("Generating roadgraphs...")
    roadgraph = generate_roadgraph(
        key=key_roadgraph, 
        batch_size=batch_size, 
        num_lanes=3, 
        lane_spacing=4.0, 
        num_points=100
    )
    logging.info("Roadgraphs generated.")

    # Generate agents
    logging.info("Generating agents...")
    agents = generate_agents(
        key=key_agents, 
        roadgraph=roadgraph, 
        num_objects=32
    )
    logging.info("Agents generated.")

    # Transform with respect to the first agent
    logging.info("Transform with respect to the first agent...")
    # agents, roadgraph = transform(agents, roadgraph)
    logging.info("Transform completed.")

    return roadgraph, agents


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate and save roadgraphs and agents.")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for generation.")
    parser.add_argument(
        "--output-dir", type=str, default="/workspace/data/processed", 
        help="Directory to save the generated files.")
    parser.add_argument(
        "--debug", action='store_true', help="Show a random batch visualization.")
    args = parser.parse_args()

    # Generate data
    logging.info(f"Starting batch generation with batch size: {args.batch_size}")
    start_time = time.time()
    roadgraph, agents = generate_batch(batch_size=args.batch_size)
    elapsed_time = time.time() - start_time
    logging.info(f"Batch generation completed in {elapsed_time:.2f} seconds.")

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Save the generated data
    np.save(os.path.join(save_dir, "roadgraph.npy"), np.array(roadgraph))
    np.save(os.path.join(save_dir, "agents.npy"), np.array(agents))
    logging.info(f"Data saved to: {save_dir}")

    if args.debug:
        # Plot the first batch
        key = random.PRNGKey(int(time.time()))
        batch_idx = random.randint(key, shape=(), minval=0, maxval=args.batch_size-1)
        fig = visualize(roadgraph, agents, batch_index=batch_idx)

        # Save the figure to a file instead of showing it
        plt.savefig(os.path.join(save_dir, f"batch_{batch_idx}.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
