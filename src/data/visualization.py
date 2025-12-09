import io, json, random
import numpy as np
import webdataset as wds
import matplotlib
matplotlib.use('Agg')  # Headless 환경을 위한 백엔드 설정
import matplotlib.pyplot as plt
from PIL import Image
import math
import cv2
import os
from collections import defaultdict

def kinematic_model_trajectory(steer_seq, accel_seq, dt=0.5, wheelbase=2.7, initial_velocity=5.0):
    """Calculate vehicle trajectory using kinematic model.
    
    Args:
        steer_seq: Steering angle sequence (radians).
        accel_seq: Acceleration sequence (m/s²).
        dt: Time step (seconds).
        wheelbase: Vehicle wheelbase (meters).
        initial_velocity: Initial velocity (m/s).
        
    Returns:
        Tuple of (x_trajectory, y_trajectory, yaw_trajectory).
    """
    x, y, yaw = 0.0, 0.0, 0.0
    velocity = initial_velocity
    
    x_traj = [x]
    y_traj = [y]
    yaw_traj = [yaw]
    
    for steer, accel in zip(steer_seq, accel_seq):
        # Update velocity
        velocity = max(0., velocity + accel * dt)  # Prevent negative velocity
        
        # Update position and orientation
        yaw_rate = (velocity * math.tan(steer)) / wheelbase
        yaw += yaw_rate * dt
        
        x += velocity * math.cos(yaw) * dt
        y += velocity * math.sin(yaw) * dt
        
        x_traj.append(x)
        y_traj.append(y)
        yaw_traj.append(yaw)
    
    return np.array(x_traj), np.array(y_traj), np.array(yaw_traj)

def project_trajectory_to_image(x_traj, y_traj, yaw_traj, camera_calib, img_height=224, img_width=224):
    """Project 3D trajectory to 2D image coordinates using actual camera calibration.
    
    Args:
        x_traj: X trajectory in meters.
        y_traj: Y trajectory in meters.
        yaw_traj: Yaw trajectory in radians.
        camera_calib: Camera calibration dictionary from nuScenes.
        img_height: Image height in pixels.
        img_width: Image width in pixels.
        
    Returns:
        Tuple of (u_coords, v_coords) in image coordinates.
    """
    # Extract camera parameters
    intrinsic_matrix = np.array(camera_calib["intrinsic_matrix"])
    camera_translation = np.array(camera_calib["translation"])
    camera_rotation = np.array(camera_calib["rotation"])  # quaternion [w, x, y, z]
    original_size = camera_calib.get("original_image_size", [1600, 900])  # [width, height]
    
    # Scale intrinsic matrix for resized image
    orig_width, orig_height = original_size
    scale_x = img_width / orig_width
    scale_y = img_height / orig_height
    
    # Create scaling matrix
    scale_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])
    
    # Scale intrinsic matrix
    scaled_intrinsic = scale_matrix @ intrinsic_matrix
    
    # Convert quaternion to rotation matrix
    from pyquaternion import Quaternion
    q = Quaternion(camera_rotation)
    R_cam = q.rotation_matrix
    
    # Project to image coordinates
    u_coords = []
    v_coords = []
    
    for x, y, yaw in zip(x_traj, y_traj, yaw_traj):
        # Vehicle position in global coordinates
        point_global = np.array([x, y, 0.0])  # z=0 (ground level)
        
        # Transform to camera coordinate system
        point_cam = R_cam.T @ (point_global - camera_translation)
        
        # Project to image plane using scaled intrinsic matrix
        if point_cam[2] > 0.1:  # Point is in front of camera (with small margin)
            point_2d = scaled_intrinsic @ point_cam
            u = point_2d[0] / point_2d[2]
            v = point_2d[1] / point_2d[2]
            
            # Accept all points regardless of bounds
            u_coords.append(u)
            v_coords.append(v)
    return np.array(u_coords), np.array(v_coords)

def show_sample_from_wds(
    wds_pattern: str,
    T: int = 4,
    K: int = 6,
    max_try: int = 20
):
    ds = wds.WebDataset(wds_pattern).decode()
    # 무작위 추출
    it = iter(ds)
    sample = None
    for _ in range(max_try):
        sample = next(it, None)
        if sample is not None:
            break
    if sample is None:
        print("No sample found.")
        return

    # 키 정리
    keys = list(sample.keys())
    meta_key = "meta.json"  # WebDataset에서 키 형식이 변경됨
    meta_data = sample[meta_key]
    if isinstance(meta_data, bytes):
        meta = json.loads(meta_data.decode("utf-8"))
    else:
        meta = meta_data  # 이미 파싱된 dict
    prompt = meta.get("prompt", "")
    lat_cont = np.array(meta["labels"]["lateral_seq_cont"], dtype=np.float32)
    lon_cont = np.array(meta["labels"]["longitudinal_seq_cont"], dtype=np.float32)
    velocities = np.array(meta["labels"]["velocities"], dtype=np.float32)
    camera_calib = meta.get("camera_calibration", {})

    # 카메라 순서: [[FL, F, FR], [BL, BACK, BR]]
    # cam0=CAM_FRONT_LEFT, cam1=CAM_FRONT, cam2=CAM_FRONT_RIGHT,
    # cam3=CAM_BACK_LEFT, cam4=CAM_BACK, cam5=CAM_BACK_RIGHT
    camera_order = [0, 1, 2, 3, 4, 5]  # [FL, F, FR, BL, BACK, BR]
    
    # 이미지 로드 (t 증가 순, 카메라 순서대로)
    imgs = []
    for t in range(T):
        row = []
        for c in camera_order:
            k = [k for k in keys if k.endswith(f"img_c{c}_t{t}.jpg")]
            assert len(k) == 1, f"missing img c{c} t{t}"
            arr = np.asarray(Image.open(io.BytesIO(sample[k[0]])).convert("RGB"))
            row.append(arr)
        imgs.append(row)

    # ----- t=0 이미지와 BEV trajectory 시각화 -----
    H_img, W_img = imgs[0][0].shape[:2]
    fig_h = (H_img / 100) + 3
    fig_w = 3 * (W_img / 100) + 4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 3 + 2)  # 3칸 이미지 + 2칸 BEV

    # 궤적 계산
    x_traj, y_traj, yaw_traj = kinematic_model_trajectory(lat_cont, lon_cont, initial_velocity=velocities[0])
    
    # t=0 이미지만 표시 (2x3 배열)
    t = T-1  # t=0 (마지막 프레임)
    # 2x3 배열로 이미지 합치기
    top_row = np.concatenate([imgs[t][0], imgs[t][1], imgs[t][2]], axis=1)  # [FL, F, FR]
    bottom_row = np.concatenate([imgs[t][3], imgs[t][4], imgs[t][5]], axis=1)  # [BL, BACK, BR]
    combined_img = np.concatenate([top_row, bottom_row], axis=0)
    
    # 전방 카메라 이미지에 궤적 오버레이
    front_img = imgs[t][1].copy()  # F (전방) 카메라 - 복사본 생성
    
    # 실제 camera calibration을 사용하여 궤적 projection
    if camera_calib:
        u_coords, v_coords = project_trajectory_to_image(
            x_traj, y_traj, yaw_traj, camera_calib, 
            front_img.shape[0], front_img.shape[1]
        )
        
        # 궤적을 전방 카메라 이미지에 그리기
        if len(u_coords) > 0:
            # 궤적을 빨간색으로 표시
            for i in range(len(u_coords)-1):
                cv2.line(front_img, 
                        (int(u_coords[i]), int(v_coords[i])), 
                        (int(u_coords[i+1]), int(v_coords[i+1])), 
                        (255, 0, 0), 3)
            
            # 시작점을 녹색으로 표시
            cv2.circle(front_img, (int(u_coords[0]), int(v_coords[0])), 8, (0, 255, 0), -1)
            
            # 끝점을 파란색으로 표시
            cv2.circle(front_img, (int(u_coords[-1]), int(v_coords[-1])), 8, (0, 0, 255), -1)
            
            # 중간점들을 노란색으로 표시
            for i in range(1, len(u_coords)-1, 2):
                cv2.circle(front_img, (int(u_coords[i]), int(v_coords[i])), 4, (0, 255, 255), -1)
    
    # 수정된 전방 이미지로 다시 합치기
    top_row = np.concatenate([imgs[t][0], front_img, imgs[t][2]], axis=1)
    combined_img = np.concatenate([top_row, bottom_row], axis=0)
    
    # t=0 이미지 표시
    ax_img = fig.add_subplot(gs[0, :3])
    ax_img.imshow(combined_img)
    ax_img.axis("off")
    ax_img.set_title("t=0 (6 cameras)", fontsize=12)

    # BEV 궤적 시각화 (우측 2칸)
    ax_traj = fig.add_subplot(gs[0, 3:])
    
    # 궤적 그리기
    ax_traj.plot(x_traj, y_traj, 'r-', linewidth=2, label='Trajectory')
    ax_traj.scatter(x_traj[0], y_traj[0], c='green', s=50, label='Start (t=0)')
    ax_traj.scatter(x_traj[-1], y_traj[-1], c='red', s=50, label='End (t=12)')
    
    # 방향 화살표 추가
    for i in range(0, len(x_traj), 2):
        if i < len(x_traj) - 1:
            dx = x_traj[i+1] - x_traj[i]
            dy = y_traj[i+1] - y_traj[i]
            ax_traj.arrow(x_traj[i], y_traj[i], dx*0.5, dy*0.5, 
                         head_width=0.5, head_length=0.3, fc='blue', ec='blue', alpha=0.7)
    
    ax_traj.set_title("BEV Trajectory (10 steps ahead)")
    ax_traj.set_xlabel("X (meters)")
    ax_traj.set_ylabel("Y (meters)")
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis('equal')

    fig.suptitle(f"Prompt: {prompt}", fontsize=12)
    plt.tight_layout()
    plt.savefig('/workspace/data/train/output.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved to /workspace/data/train/output.png")
    plt.close()


def create_scene_videos(wds_pattern: str, output_dir: str = "/workspace/data/train/videos"):
    """Create videos for each scene showing all frames with trajectory overlay.
    
    Args:
        wds_pattern: WebDataset pattern.
        output_dir: Output directory for videos.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group samples by scene
    ds = wds.WebDataset(wds_pattern).decode()
    scene_samples = defaultdict(list)
    
    print("Grouping samples by scene...")
    for sample in ds:
        meta_data = sample['meta.json']
        if isinstance(meta_data, bytes):
            meta = json.loads(meta_data.decode('utf-8'))
        else:
            meta = meta_data
        
        # Extract scene token from sample key
        sample_key = sample.get('__key__', '')
        if 'scene_token' in meta:
            scene_token = meta['scene_token']
        else:
            # Extract from sample key if not in meta
            scene_token = sample_key.split('-')[0] if '-' in sample_key else 'unknown'
        
        scene_samples[scene_token].append((sample, meta))
    
    print(f"Found {len(scene_samples)} scenes")
    
    # Create video for each scene
    for scene_token, samples in scene_samples.items():
        print(f"Creating video for scene {scene_token} with {len(samples)} samples...")
        
        # Sort samples by timestamp
        samples.sort(key=lambda x: x[1].get('center_timestamp', 0))
        
        # Get first sample to determine video properties
        first_sample, first_meta = samples[0]
        
        # Get image dimensions
        img_keys = [k for k in first_sample.keys() if k.endswith('.jpg')]
        if not img_keys:
            print(f"No images found for scene {scene_token}")
            continue
            
        # Load first image to get dimensions
        first_img = Image.open(io.BytesIO(first_sample[img_keys[0]])).convert('RGB')
        img_height, img_width = first_img.size[1], first_img.size[0]  # PIL uses (width, height)
        
        # Create 2x3 combined image dimensions + BEV
        combined_height = img_height * 2
        combined_width = img_width * 4  # 3 for cameras + 1 for BEV
        
        # Setup video writer
        video_path = os.path.join(output_dir, f"{scene_token}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 2.0  # 2 FPS (0.5 second intervals)
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (combined_width, combined_height))
        
        if not video_writer.isOpened():
            print(f"Failed to create video writer for {scene_token}")
            continue
        
        # Process each sample in the scene
        for sample, meta in samples:
            # Load images for this sample
            imgs = []
            for t in range(4):  # T=4
                row = []
                for c in range(6):  # K=6
                    img_key = f"img_c{c}_t{t}.jpg"
                    if img_key in sample:
                        img = Image.open(io.BytesIO(sample[img_key])).convert('RGB')
                        row.append(np.array(img))
                    else:
                        # Create black image if missing
                        row.append(np.zeros((img_height, img_width, 3), dtype=np.uint8))
                imgs.append(row)
            
            # Create 2x3 combined image for t=0 (last frame)
            t = 3  # t=0 (last frame)
            top_row = np.concatenate([imgs[t][0], imgs[t][1], imgs[t][2]], axis=1)  # [FL, F, FR]
            bottom_row = np.concatenate([imgs[t][3], imgs[t][4], imgs[t][5]], axis=1)  # [BL, BACK, BR]
            camera_img = np.concatenate([top_row, bottom_row], axis=0)
            
            # Add trajectory overlay to front camera
            camera_calib = meta.get('camera_calibration', {})
            if camera_calib:
                lat_cont = np.array(meta['labels']['lateral_seq_cont'])
                lon_cont = np.array(meta['labels']['longitudinal_seq_cont'])
                velocities = np.array(meta['labels']['velocities'], dtype=np.float32)

                x_traj, y_traj, yaw_traj = kinematic_model_trajectory(lat_cont, lon_cont, initial_velocity=velocities[0])
                
                front_img = imgs[t][1].copy()  # F (전방) 카메라
                u_coords, v_coords = project_trajectory_to_image(
                    x_traj, y_traj, yaw_traj, camera_calib, 
                    front_img.shape[0], front_img.shape[1]
                )
                
                # Draw trajectory
                if len(u_coords) > 0:
                    for i in range(len(u_coords)-1):
                        cv2.line(front_img, 
                                (int(u_coords[i]), int(v_coords[i])), 
                                (int(u_coords[i+1]), int(v_coords[i+1])), 
                                (0, 0, 255), 2)
                        cv2.circle(front_img, (int(u_coords[i]), int(v_coords[i])), 3, (0, 0, 255), -1)
                    
                    cv2.circle(front_img, (int(u_coords[-1]), int(v_coords[-1])), 3, (0, 0, 255), -1)
                
                # Update camera image with trajectory
                top_row = np.concatenate([imgs[t][0], front_img, imgs[t][2]], axis=1)
                camera_img = np.concatenate([top_row, bottom_row], axis=0)
            
            # Create BEV trajectory plot
            bev_img = create_bev_trajectory_plot(meta, img_height * 2, img_width)
            
            # Combine camera image and BEV
            combined_img = np.concatenate([camera_img, bev_img], axis=1)
            
            # Convert to BGR for OpenCV
            combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
            
            # Write frame to video
            video_writer.write(combined_img_bgr)
        
        video_writer.release()
        print(f"✅ Video saved: {video_path}")
    
    print(f"✅ All scene videos created in {output_dir}")


def create_bev_trajectory_plot(meta: dict, height: int, width: int) -> np.ndarray:
    """Create BEV trajectory plot as numpy array.
    
    Args:
        meta: Sample metadata containing trajectory data
        height: Height of the BEV plot
        width: Width of the BEV plot
        
    Returns:
        BEV trajectory plot as RGB numpy array
    """
    # Create figure with specified size
    fig, ax = plt.subplots(1, 1, figsize=(width/100, height/100), dpi=100)
    
    # Get trajectory data
    lat_cont = np.array(meta['labels']['lateral_seq_cont'])
    lon_cont = np.array(meta['labels']['longitudinal_seq_cont'])
    velocities = np.array(meta['labels']['velocities'], dtype=np.float32)
    
    # Generate trajectory
    x_traj, y_traj, yaw_traj = kinematic_model_trajectory(lat_cont, lon_cont, initial_velocity=velocities[0])
    
    # Rotate coordinates 90 degrees counterclockwise: (x, y) -> (-y, x)
    # This makes x-axis point up (forward) and y-axis point left
    x_rotated = -y_traj  # x becomes -y (forward direction)
    y_rotated = x_traj   # y becomes x (left direction)
    
    # Rotate yaw angles by 90 degrees counterclockwise
    yaw_rotated = yaw_traj + np.pi/2
    
    # Fixed plot range for consistent visualization
    x_range = 40.0
    y_range = 100.0
    
    # Set fixed axis limits
    ax.set_xlim(-x_range * 0.5, x_range * 0.5)
    ax.set_ylim(-y_range * 0.2, y_range * 0.8)
    
    # Plot trajectory
    ax.plot(x_rotated, y_rotated, 'bo-', linewidth=1, markersize=2, label='Trajectory')
    
    # Plot vehicle orientation arrows
    for i in range(0, len(x_rotated), 2):  # Every 2nd point
        dx = 0.5 * np.cos(yaw_rotated[i])
        dy = 0.5 * np.sin(yaw_rotated[i])
        ax.arrow(x_rotated[i], y_rotated[i], dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    ax.set_title('Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Convert plot to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    # Resize to target dimensions
    buf_resized = cv2.resize(buf, (width, height))
    
    return buf_resized


if __name__ == "__main__":
    show_sample_from_wds("/workspace/data/train/00000.tar", T=4, K=6)
    create_scene_videos("/workspace/data/train/00000.tar")