import numpy as np

def get_2d_corners(locs: np.ndarray, dims: np.ndarray, rots: np.ndarray) -> np.ndarray:
    """Get the corners of a bounding box.

    Args:
        locs (np.ndarray): The location of the bounding box. (N, 3)
        dims (np.ndarray): The dimensions of the bounding box. (N, 3)
        rots (np.ndarray): The rotation of the bounding box. (N, 1)

    Returns:
        np.ndarray: The corners of the bounding box. (N, 8, 3)
    """
    locs = np.asarray(locs)
    dims = np.asarray(dims)
    rots = np.asarray(rots)

    # --- locs / dims 은 항상 (N, 3) 로 맞추기 ---
    locs = locs.reshape(-1, 3)
    dims = dims.reshape(-1, 3)
    N = locs.shape[0]

    # --- rots 처리: 다양한 shape를 허용 ---
    if rots.ndim == 1:
        # (N,) or (N*3,)
        if rots.size == N:
            yaw = rots
        elif rots.size == 3 * N:
            yaw = rots.reshape(-1, 3)[:, -1]  # 마지막 컬럼을 yaw로 사용
        else:
            raise ValueError(
                f"rots length {rots.size} is incompatible with N={N}. "
                "Expected N or 3*N elements."
            )
    elif rots.ndim == 2:
        if rots.shape[0] != N:
            raise ValueError(
                f"rots first dim {rots.shape[0]} != N={N}. "
                "locs, dims, rots must describe the same number of objects."
            )
        if rots.shape[1] == 1:
            yaw = rots[:, 0]
        else:
            # (N, 3) 같은 경우: 마지막 컬럼을 yaw로 사용
            yaw = rots[:, -1]
    else:
        raise ValueError("rots must be 1D or 2D array.")

    # --- 중심 및 크기 ---
    centers_xy = locs[:, :2]   # (N, 2)
    l = dims[:, 0]             # length
    w = dims[:, 1]             # width

    # --- 박스 로컬 좌표계에서의 코너 (회전 전) ---
    # 순서는 [-l/2, -w/2], [-l/2, w/2], [l/2, w/2], [l/2, -w/2]
    local_corners = np.array([
        [-0.5, -0.5],
        [-0.5,  0.5],
        [ 0.5,  0.5],
        [ 0.5, -0.5],
    ])  # (4, 2)

    # 각 박스별 (l, w) 스케일 적용 → (N, 4, 2)
    scale = np.stack([l, w], axis=-1)[:, None, :]   # (N, 1, 2)
    corners_local_scaled = scale * local_corners[None, :, :]  # (N, 4, 2)

    # --- yaw 회전 행렬 (z-축 기준) ---
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # rot_mats[n] =
    # [[cos, -sin],
    #  [sin,  cos]]
    rot_mats = np.stack([
        np.stack([cos_yaw, -sin_yaw], axis=-1),
        np.stack([sin_yaw,  cos_yaw], axis=-1),
    ], axis=1)  # (N, 2, 2)

    # --- 회전 적용 ---
    # corners_local_scaled: (N, 4, 2) → indices (n, k, i)
    # rot_mats:           (N, 2, 2) → indices (n, i, j)
    # 결과:                (N, 4, 2) → (n, k, j)
    corners_rotated = np.einsum('nki,nij->nkj', corners_local_scaled, rot_mats)

    # --- 트랜슬레이션 (center x, y) 적용 ---
    corners_world = corners_rotated + centers_xy[:, None, :]  # (N, 4, 2)

    return corners_world