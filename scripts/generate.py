#!/usr/bin/env python3

import os
import argparse
import time
from datetime import datetime
import logging
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt

from ml4d.data.agents import generate_agents
from ml4d.data.roadgraph import generate_roadgraph
from ml4d.data.visualize import visualize
from ml4d.data.teacher import generate_init_policy, generate_keeping_policy
from ml4d.sim.agent.policy import find_nearest_lane, find_front_vehicle
from ml4d.utils.transform import transform_roadgraph, transform_agents
from ml4d.utils.unit import kph2mps, deg2rad


WHEELBASE = 2.5

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


def simulate(agents: jax.Array,
             policy: jax.Array,
             dt: float = 0.1) -> jax.Array:
    """
    Update agent states for a single time step based on the given teacher policy.

    Args:
        agents (jax.Array): A 3D array of shape (batch_size, num_objects, state_dim+1).
            Example state_dim fields might be 
            [x, y, cos_heading, sin_heading, speed, length, width, valid].
            The last column is 'valid' (1 or 0).
        policy (jax.Array): A 3D array of shape (batch_size, num_objects, 3),
            where each row is [delta, accel, valid].
        dt (float, optional): Time step for the update. Defaults to 0.1 (seconds).

    Returns:
        jax.Array: The updated agents array (same shape as input).
    """
    # Unpack relevant fields
    x = agents[..., 0]
    y = agents[..., 1]
    cos_h = agents[..., 2]
    sin_h = agents[..., 3]
    heading = jnp.arctan2(sin_h, cos_h)
    speed = agents[..., 4]

    # Teacher policy outputs
    delta = policy[..., 0]
    accel = policy[..., 1]

    # Simple bicycle model heading update
    # Update position (x, y)
    new_x = x + speed * cos_h * dt
    new_y = y + speed * sin_h * dt

    # Recompute cos and sin of new heading
    new_heading = heading + (speed / WHEELBASE) * jnp.tan(delta) * dt
    new_cos_h = jnp.cos(new_heading)
    new_sin_h = jnp.sin(new_heading)
    
    # Update speed
    new_speed = jnp.maximum(speed + accel * dt, 0.)

    # Store the updated states back into the agent array
    new_agents = agents
    new_agents = new_agents.at[..., 0].set(new_x)
    new_agents = new_agents.at[..., 1].set(new_y)
    new_agents = new_agents.at[..., 2].set(new_cos_h)
    new_agents = new_agents.at[..., 3].set(new_sin_h)
    new_agents = new_agents.at[..., 4].set(new_speed)
    
    return new_agents


def rollout(roadgraph: jax.Array,
            agents: jax.Array,
            policy: jax.Array,
            horizon: int = 30,
            dt: float = 0.2) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Perform a multi-step rollout using the teacher policy for a given roadgraph and agents.

    Args:
        roadgraph (jax.Array): A 4D array of shape (batch_size, num_lanes, num_points, 2)
            describing the road layout.
        agents (jax.Array): A 3D array of shape (batch_size, num_objects, state_dim+1)
            containing agent states.
        horizon (int, optional): Number of rollout steps. Defaults to 10.
        dt (float, optional): Timestep for each update. Defaults to 0.1.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]:
            A tuple of (roadgraphs, agents, policy), each with an extra time dimension.
            Shapes (num_objects = N):
                - roadgraphs: (horizon, num_lanes, num_points, 3)
                - agents: (horizon, N, state_dim+1)
                - policy: (horizon, N, 3)
    """
    # Buffers to store the rollout data at each time step
    agents_buffer = []
    policy_buffer = []
    
    # Find lane index and front vehicle
    lane_indices = find_nearest_lane(roadgraph, agents)
    front_incides = find_front_vehicle(agents, lane_indices)
    
    for _ in tqdm(range(horizon), desc='Planning horizon', total=horizon, leave=False):
        # Store data
        agents_buffer.append(agents)
        policy_buffer.append(policy)
        
        # Update agent states based on the policy
        agents = simulate(agents, policy, dt=dt)
        
        # Update teacher policy (delta, accel)
        policy = generate_keeping_policy(
            roadgraph=roadgraph, 
            agents=agents,
            lane_indices=lane_indices,
            front_indices=front_incides,
            lookahead_time=1.0,
            wheelbase=WHEELBASE,
            speed_limit=kph2mps(50.),
        )

    # Stack along a new time axis
    rollout_agents = jnp.stack(agents_buffer, axis=1)
    rollout_policy = jnp.stack(policy_buffer, axis=1)
        
    return rollout_agents, rollout_policy


def generate_batch(batch_size: int = 128,
                   horizon: int = 30,
                   dt: float = 0.2) -> tuple[jax.Array, jax.Array]:
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
    logging.info("Initialize...")
    
    # Random generator setup
    key = random.PRNGKey(42)
    key_roadgraph, key_agents, key_policy = random.split(key, 3)

    # Generate roadgraph
    init_roadgraph = generate_roadgraph(
        key=key_roadgraph, 
        batch_size=batch_size, 
        num_lanes=3, 
        lane_spacing=4.0, 
        num_points=100,
        position=(-2., 2.),
        heading=(deg2rad(-10), deg2rad(10)),
        curvature=(-0.005, 0.005),
    )

    # Generate initial agents
    init_agents = generate_agents(
        key=key_agents, 
        batch_size=batch_size,
        roadgraph=init_roadgraph, 
        num_objects=32,
        noise=(1.0, 1.0, deg2rad(5.0)),
        speed=(kph2mps(20.), kph2mps(30.)),
        length=(4.8, 5.2),
        width=(1.8, 2.2),
    )
    
    # Generate initial policy
    init_policy = generate_init_policy(
        key=key_policy,
        agents=init_agents,
        delta=(-deg2rad(0.), deg2rad(0.)),
        accel=(-0., 0.),
    )
    
    # Transform data
    roadgraph = transform_roadgraph(init_roadgraph, init_agents)
    agents = transform_agents(init_agents)
    
    # Generate batched rollout data
    agents, policy = rollout(
        roadgraph=roadgraph,
        agents=agents,
        policy=init_policy,
        horizon=horizon,
        dt=dt,
    )
    
    return roadgraph, agents, policy


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate and save roadgraphs and agents.")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for generation.")
    parser.add_argument(
        "--horizon", type=int, default=30, help="Planning horizon.")
    parser.add_argument(
        "--dt", type=float, default=0.2, help="Step time for planning.")
    parser.add_argument(
        "--output-dir", type=str, default="/workspace/data/processed", 
        help="Directory to save the generated files.")
    parser.add_argument(
        "--debug", action='store_true', help="Show a random batch visualization.")
    args = parser.parse_args()

    # Generate data
    logging.info(f"Starting data generation with batch size: {args.batch_size}")
    start_time = time.time()
    roadgraph, agents, policy = generate_batch(
        batch_size=args.batch_size,
        horizon=args.horizon,
        dt=args.dt
    )
    elapsed_time = time.time() - start_time
    logging.info(f"Data generation completed in {elapsed_time:.2f} seconds.")

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Save the generated data
    np.save(os.path.join(save_dir, "roadgraph.npy"), np.array(roadgraph))
    np.save(os.path.join(save_dir, "agents.npy"), np.array(agents))
    np.save(os.path.join(save_dir, "policy.npy"), np.array(policy))
    logging.info(f"Data saved to: {save_dir}")

    if args.debug:
        # Plot the first batch
        key = random.PRNGKey(int(time.time()))
        batch_idx = random.randint(key, shape=(), minval=0, maxval=args.batch_size-1)
        batch_idx = 819
        fig = visualize(
            roadgraph=roadgraph,
            agents=agents,
            batch_idx=batch_idx,
        )

        # Save the figure to a file instead of showing it
        plt.savefig(os.path.join(save_dir, f"batch_{batch_idx}.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
