import jax
import jax.numpy as jnp

from ml4d.sim.agent.policy import (
    find_nearest_lane, find_front_vehicle)
from ml4d.sim.agent.idm import idm
from ml4d.sim.agent.pure_pursuit import pure_pursuit
from ml4d.utils.unit import kph2mps, deg2rad, mod2pi


def generate_keeping_policy(roadgraph: jax.Array,
                            agents: jax.Array) -> jax.Array:
    """
    Generate control input for lane keeping.

    Args:
        roadgraph (jax.Array): Roadgraph 
            shape: (batch_size, num_lanes, num_points, 2)
        agents (jax.Array): Agent states 
            shape: (batch_size, num_objects, state_dim+1)
            마지막 컬럼은 valid 플래그 (1 또는 0)

    Returns:
        jax.Array: (batch_size, num_objects, input_dim+1)
        여기서는 (batch_size, num_objects, 3)이라 가정.
        순서: [delta(조향), accel(가속), valid]
    """
    # Find nearest lane for each agent
    lane_idx = find_nearest_lane(roadgraph, agents)  # (batch, num_objects)
    
    # Find the front vehicle index
    front_idx = find_front_vehicle(agents, lane_idx) # (batch, num_objects)

    # 유효한 lane_idx인 경우 valid=1, 아니면 valid=0
    valid = (lane_idx != -1).astype(jnp.float32)  # (batch, num_objects)

    # 조향각(delta)은 일단 0으로 가정
    delta = jnp.zeros_like(lane_idx, dtype=jnp.float32)

    # 앞 차량 존재 여부에 따라 accel 결정
    # 앞 차량 있으면 accel=0, 없으면 accel=1
    has_front = (front_idx != -1)
    accel = jnp.where(has_front, 0.0, 1.0).astype(jnp.float32)

    # invalid한 에이전트의 입력은 전부 0
    delta = delta * valid
    accel = accel * valid

    # 최종 스택
    delta = delta[..., None]  # (batch, num_objects, 1)
    accel = accel[..., None]  # (batch, num_objects, 1)
    valid = valid[..., None]  # (batch, num_objects, 1)

    policy = jnp.concatenate([delta, accel, valid], axis=-1)  # (batch, num_objects, 3)
    return policy
    
    