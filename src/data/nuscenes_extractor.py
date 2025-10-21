import os
import cv2
import json
import math
import bisect
import numpy as np
from typing import List, Dict, Tuple, Optional

import webdataset as wds
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion


def quat_to_yaw(quaternion: List[float]) -> float:
    """Convert quaternion to yaw angle.
    
    Args:
        quaternion: Quaternion as [w, x, y, z].
        
    Returns:
        Yaw angle in radians.
    """
    q = Quaternion(quaternion)
    return float(q.yaw_pitch_roll[0])


def normalize_angle_diff(angle1: float, angle2: float) -> float:
    """Calculate normalized angle difference between two angles.
    
    Args:
        angle1: First angle in radians.
        angle2: Second angle in radians.
        
    Returns:
        Normalized angle difference in [-π, π].
    """
    diff = angle2 - angle1
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff


def encode_image_to_jpg(image: np.ndarray, size: Tuple[int, int], quality: int = 90) -> Optional[bytes]:
    """Encode BGR image to JPEG bytes.
    
    Args:
        image: BGR image array.
        size: Target size as (height, width).
        quality: JPEG quality (1-100).
        
    Returns:
        JPEG encoded bytes or None if failed.
    """
    height, width = size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    success, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if success else None


def quantize_value(value: float, bin_edges: np.ndarray) -> int:
    """Quantize continuous value to discrete bin.
    
    Args:
        value: Continuous value to quantize.
        bin_edges: Array of bin edges (length N+1 for N bins).
        
    Returns:
        Bin index (0 to N-1).
    """
    return int(np.clip(np.searchsorted(bin_edges, value, side="right") - 1, 0, len(bin_edges) - 2))


def create_uniform_bins(min_val: float, max_val: float, num_bins: int) -> np.ndarray:
    """Create uniform bin edges.
    
    Args:
        min_val: Minimum value.
        max_val: Maximum value.
        num_bins: Number of bins.
        
    Returns:
        Array of bin edges.
    """
    return np.linspace(min_val, max_val, num_bins + 1, dtype=np.float32)


def find_nearest_timestamp_index(timestamps: List[int], target: int) -> int:
    """Find index of nearest timestamp in sorted list.
    
    Args:
        timestamps: Sorted list of timestamps.
        target: Target timestamp.
        
    Returns:
        Index of nearest timestamp.
    """
    idx = bisect.bisect_left(timestamps, target)
    if idx == 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    
    before = timestamps[idx - 1]
    after = timestamps[idx]
    return idx - 1 if (target - before) <= (after - target) else idx


class NuScenesExtractor:
    """Extract multi-camera sequences and action labels from nuScenes dataset."""
    
    def __init__(
        self,
        nuscenes_root: str,
        out_dir: str,
        split: str = "train",
        input_frames: int = 4,
        input_interval: float = 0.5,
        output_steps: int = 10,
        output_interval: float = 0.5,
        img_size: Tuple[int, int] = (224, 224),
        steer_range: Tuple[float, float] = (-0.6, 0.6),
        accel_range: Tuple[float, float] = (-6.0, 6.0),
        wheelbase: float = 2.7,
        jpg_quality: int = 90,
        time_tolerance_ms: int = 120,
    ):
        """Initialize NuScenes extractor.
        
        Args:
            nuscenes_root: Path to nuScenes dataset.
            out_dir: Output directory for shards.
            split: Dataset split ('train', 'val', 'test').
            input_frames: Number of input frames (T).
            input_interval: Time interval between input frames in seconds.
            output_steps: Number of output action steps (H).
            output_interval: Time interval between output steps in seconds.
            img_size: Image size as (height, width).
            steer_range: Steering angle range in radians.
            accel_range: Acceleration range in m/s².
            wheelbase: Vehicle wheelbase in meters.
            jpg_quality: JPEG compression quality.
            time_tolerance_ms: Time tolerance for frame matching in milliseconds.
        """
        self.nuscenes_root = nuscenes_root
        self.out_dir = out_dir
        self.split = split
        self.T = input_frames
        self.dt_in_sec = input_interval
        self.H = output_steps
        self.dt_out_sec = output_interval
        self.img_size = img_size
        self.wheelbase = wheelbase
        self.jpg_quality = jpg_quality
        self.time_tolerance_us = time_tolerance_ms * 1000
        
        self.cameras = [
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
        ]
        
        # Create bin edges for quantization
        self.steer_edges = create_uniform_bins(steer_range[0], steer_range[1], 255)
        self.accel_edges = create_uniform_bins(accel_range[0], accel_range[1], 255)
        
        # Initialize nuScenes
        self.nusc = NuScenes(version="v1.0-mini", dataroot=nuscenes_root, verbose=True)
        self.split_scenes = set(create_splits_scenes()[split])
        self.scenes = [s for s in self.nusc.scene if s["name"] in self.split_scenes]
    
    def extract_shards(self, max_samples_per_shard: int = 5000, verbose_every: int = 1000):
        """Extract and save data shards.
        
        Args:
            max_samples_per_shard: Maximum samples per shard file.
            verbose_every: Print progress every N samples.
        """
        # Setup output directory and shard writer
        split_dir = os.path.join(self.out_dir, self.split)
        os.makedirs(split_dir, exist_ok=True)
        shard_pattern = os.path.join(split_dir, "%05d.tar")
        sink = wds.ShardWriter(shard_pattern, maxcount=max_samples_per_shard, maxsize=1024*1024*1024)
        
        seq_id = 0
        skipped = 0
        
        # Process each scene
        for scene in self.scenes:
            seq_id, skipped = self._process_scene(scene, sink, seq_id, skipped, verbose_every)
        
        sink.close()
        print(f"[DONE] split={self.split}, samples={seq_id}, skipped={skipped}, shards={shard_pattern}")
    
    def _process_scene(self, scene: Dict, sink: wds.ShardWriter, seq_id: int, skipped: int, verbose_every: int) -> Tuple[int, int]:
        """Process a single scene and extract sequences.
        
        Args:
            scene: nuScenes scene object.
            sink: WebDataset shard writer.
            seq_id: Current sequence ID.
            skipped: Current skipped count.
            verbose_every: Print progress every N samples.
            
        Returns:
            Updated (seq_id, skipped) counts.
        """
        # Build timeline of samples
        timeline = self._build_timeline(scene)
        if not timeline:
            return seq_id, skipped
        
        # Get reference camera timestamps
        ref_timestamps = self._get_reference_timestamps(timeline)
        if len(ref_timestamps) < (self.T + self.H):
            return seq_id, skipped

        # Process each center timestamp
        dt_in_us = int(round(self.dt_in_sec * 1e6))
        dt_out_us = int(round(self.dt_out_sec * 1e6))
        
        for center_ts in ref_timestamps:
            seq_id, skipped = self._process_sequence(
                center_ts, timeline, ref_timestamps, sink, seq_id, skipped, verbose_every
            )
        
        return seq_id, skipped
    
    def _build_timeline(self, scene: Dict) -> List[Dict]:
        """Build timeline of samples for a scene.
        
        Args:
            scene: nuScenes scene object.
            
        Returns:
            List of sample objects in chronological order.
        """
        timeline = []
        token = scene["first_sample_token"]
        while token:
            sample = self.nusc.get("sample", token)
            timeline.append(sample)
            token = sample["next"]
        return timeline
    
    def _get_reference_timestamps(self, timeline: List[Dict]) -> List[int]:
        """Get reference camera timestamps from timeline.
        
        Args:
            timeline: List of sample objects.
            
        Returns:
            Sorted list of timestamps in microseconds.
        """
        ref_cam = self.cameras[0]  # CAM_FRONT
        timestamps = []
        for sample in timeline:
            sd_token = sample["data"].get(ref_cam, None)
            if sd_token is None:
                continue
            sd = self.nusc.get("sample_data", sd_token)
            timestamps.append(sd["timestamp"])
        timestamps.sort()
        return timestamps
    
    def _process_sequence(
        self, 
        center_ts: int, 
        timeline: List[Dict], 
        ref_timestamps: List[int], 
        sink: wds.ShardWriter, 
        seq_id: int, 
        skipped: int, 
        verbose_every: int
    ) -> Tuple[int, int]:
        """Process a single sequence centered at given timestamp.
        
        Args:
            center_ts: Center timestamp for the sequence.
            timeline: List of sample objects.
            ref_timestamps: Reference timestamps.
            sink: WebDataset shard writer.
            seq_id: Current sequence ID.
            skipped: Current skipped count.
            verbose_every: Print progress every N samples.
            
        Returns:
            Updated (seq_id, skipped) counts.
        """
        # Calculate target timestamps
        dt_in_us = int(round(self.dt_in_sec * 1e6))
        dt_out_us = int(round(self.dt_out_sec * 1e6))
        
        input_targets = [center_ts - (self.T - 1 - t) * dt_in_us for t in range(self.T)]
        output_targets = [center_ts + k * dt_out_us for k in range(self.H)]
        
        # Find nearest timestamps
        input_indices = [find_nearest_timestamp_index(ref_timestamps, ts) for ts in input_targets]
        output_indices = [find_nearest_timestamp_index(ref_timestamps, ts) for ts in output_targets]
        
        # Check time tolerance
        if any(abs(ref_timestamps[i] - ts) > self.time_tolerance_us for i, ts in zip(input_indices, input_targets)):
            return seq_id, skipped + 1
        if any(abs(ref_timestamps[i] - ts) > self.time_tolerance_us for i, ts in zip(output_indices, output_targets)):
            return seq_id, skipped + 1
        
        # Load images
        images = self._load_images(timeline, ref_timestamps, input_indices)
        if not images:
            return seq_id, skipped + 1
        
        # Generate action labels
        actions = self._generate_actions(timeline, ref_timestamps, output_indices)
        if not actions:
            return seq_id, skipped + 1
        
        # Save sample
        self._save_sample(sink, seq_id, center_ts, images, actions, timeline, ref_timestamps)
        seq_id += 1
        
        if verbose_every and (seq_id % verbose_every == 0):
            print(f"[{self.split}] written {seq_id} | skipped {skipped}")
        
        return seq_id, skipped
    
    def _load_images(self, timeline: List[Dict], ref_timestamps: List[int], input_indices: List[int]) -> Optional[Dict]:
        """Load and encode images for input sequence.
        
        Args:
            timeline: List of sample objects.
            ref_timestamps: Reference timestamps.
            input_indices: Indices for input frames.
            
        Returns:
            Dictionary of encoded images or None if failed.
        """
        images = {}
        ref_cam = self.cameras[0]
        
        for t, idx in enumerate(input_indices):
            sample = timeline[idx]
            
            for c, cam in enumerate(self.cameras):
                sd_token = sample["data"].get(cam, None)
                if sd_token is None:
                    return None
                
                sd = self.nusc.get("sample_data", sd_token)
                img_path = os.path.join(self.nuscenes_root, sd["filename"])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                
                # Flip rear-facing cameras horizontally
                if cam in ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
                    img = cv2.flip(img, 1)  # 1 = horizontal flip
                
                encoded = encode_image_to_jpg(img, self.img_size, self.jpg_quality)
                if encoded is None:
                    return None
                
                images[(c, t)] = encoded
        
        return images
    
    def _generate_actions(self, timeline: List[Dict], ref_timestamps: List[int], output_indices: List[int]) -> Optional[Dict]:
        """Generate action sequences from ego motion.
        
        Args:
            timeline: List of sample objects.
            ref_timestamps: Reference timestamps.
            output_indices: Indices for output frames.
            
        Returns:
            Dictionary of action sequences or None if failed.
        """
        ref_cam = self.cameras[0]
        lateral_continuous = []
        longitudinal_continuous = []
        velocities = []
        
        for k, idx in enumerate(output_indices):
            current_sample = timeline[idx]
            
            # Use previous output frame for consistent 0.5s intervals
            if k > 0:
                prev_idx = output_indices[k - 1]
                prev_sample = timeline[prev_idx]
            else:
                # For first frame, use previous timeline frame
                prev_idx = max(idx - 1, 0)
                prev_sample = timeline[prev_idx]
            
            # Get ego states
            current_state = self._get_ego_state(current_sample, ref_cam)
            prev_state = self._get_ego_state(prev_sample, ref_cam)
            
            if current_state is None or prev_state is None:
                return None
            
            xm1, ym1, yawm1, tm1 = prev_state
            x0, y0, yaw0, t0 = current_state
            
            dt = max(t0 - tm1, 1e-3)
            velocity = math.hypot(x0 - xm1, y0 - ym1) / dt
            yaw_rate = normalize_angle_diff(yawm1, yaw0) / dt
            
            # Calculate acceleration using consistent intervals
            if k > 1:
                # Use previous two output frames for acceleration calculation
                prev2_idx = output_indices[k - 2]
                prev2_sample = timeline[prev2_idx]
                prev2_state = self._get_ego_state(prev2_sample, ref_cam)
                if prev2_state is None:
                    return None
                xm2, ym2, _, tm2 = prev2_state
                dt_prev = max(tm1 - tm2, 1e-3)
                velocity_prev = math.hypot(xm1 - xm2, ym1 - ym2) / dt_prev
            else:
                velocity_prev = velocity
            
            acceleration = (velocity - velocity_prev) / dt
            steering = math.atan2(self.wheelbase * yaw_rate, max(velocity, 1e-3))
            
            lateral_continuous.append(steering)
            longitudinal_continuous.append(acceleration)
            velocities.append(velocity)
        
        # Quantize actions
        lateral_bins = [quantize_value(x, self.steer_edges) for x in lateral_continuous]
        longitudinal_bins = [quantize_value(x, self.accel_edges) for x in longitudinal_continuous]
        
        return {
            "lateral_continuous": lateral_continuous,
            "lateral_bins": lateral_bins,
            "longitudinal_continuous": longitudinal_continuous,
            "longitudinal_bins": longitudinal_bins,
            "blinker": 0,
            "velocities": velocities
        }
    
    def _get_ego_state(self, sample: Dict, ref_cam: str) -> Optional[Tuple[float, float, float, float]]:
        """Extract ego vehicle state from sample.
        
        Args:
            sample: nuScenes sample object.
            ref_cam: Reference camera name.
            
        Returns:
            Tuple of (x, y, yaw, timestamp_sec) or None if failed.
        """
        try:
            sd_token = sample["data"][ref_cam]
            sd = self.nusc.get("sample_data", sd_token)
            ego_pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
            
            x, y, _ = ego_pose["translation"]
            yaw = quat_to_yaw(ego_pose["rotation"])
            timestamp_sec = ego_pose["timestamp"] / 1e6
            
            return (x, y, yaw, timestamp_sec)
        except:
            return None
    
    def _save_sample(self, sink: wds.ShardWriter, seq_id: int, center_ts: int, images: Dict, actions: Dict, timeline: List[Dict], ref_timestamps: List[int]):
        """Save a single sample to WebDataset.
        
        Args:
            sink: WebDataset shard writer.
            seq_id: Sequence ID.
            center_ts: Center timestamp.
            images: Dictionary of encoded images.
            actions: Dictionary of action sequences.
            timeline: List of sample objects.
            ref_timestamps: Reference timestamps.
        """
        key = f"seq-{seq_id:06d}"
        
        # Get camera calibration for CAM_FRONT
        camera_calib = self._get_camera_calibration(timeline[ref_timestamps.index(center_ts)])
        
        # Get scene token from the center timestamp sample
        center_sample = timeline[ref_timestamps.index(center_ts)]
        scene_token = center_sample["scene_token"]
        
        meta = {
            "dataset": "nuScenes",
            "scene_token": scene_token,
            "center_timestamp": int(center_ts),
            "cameras": self.cameras,
            "T": self.T,
            "H": self.H,
            "dt_in_sec": float(self.dt_in_sec),
            "dt_out_sec": float(self.dt_out_sec),
            "img_size": [int(self.img_size[0]), int(self.img_size[1])],
            "camera_calibration": camera_calib,
            "edges": {
                "steer_edges": [float(e) for e in self.steer_edges],
                "accel_edges": [float(e) for e in self.accel_edges],
            },
            "labels": {
                "lateral_seq_cont": [float(x) for x in actions["lateral_continuous"]],
                "lateral_seq_bin": [int(b) for b in actions["lateral_bins"]],
                "longitudinal_seq_cont": [float(x) for x in actions["longitudinal_continuous"]],
                "longitudinal_seq_bin": [int(b) for b in actions["longitudinal_bins"]],
                "blinker": int(actions["blinker"]),
                "velocities": [float(v) for v in actions["velocities"]]
            }
        }
        
        sample_data = {
            "__key__": key,
            "meta.json": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
        }
        
        for t in range(self.T):
            for c in range(len(self.cameras)):
                sample_data[f"img_c{c}_t{t}.jpg"] = images[(c, t)]
        
        sink.write(sample_data)
    
    def _get_camera_calibration(self, sample: Dict) -> Dict:
        """Extract camera calibration parameters for CAM_FRONT.
        
        Args:
            sample: nuScenes sample object.
            
        Returns:
            Dictionary containing camera calibration parameters.
        """
        try:
            # Get CAM_FRONT sample data
            front_cam = "CAM_FRONT"
            sd_token = sample["data"][front_cam]
            sd = self.nusc.get("sample_data", sd_token)
            
            # Get calibrated sensor
            calib_token = sd["calibrated_sensor_token"]
            calib = self.nusc.get("calibrated_sensor", calib_token)
            
            # Extract intrinsic matrix and other parameters
            intrinsic = calib["camera_intrinsic"]
            
            return {
                "camera_name": front_cam,
                "intrinsic_matrix": intrinsic,
                "translation": calib["translation"],
                "rotation": calib["rotation"],
                "sensor_token": calib["sensor_token"],
                "original_image_size": [sd["width"], sd["height"]]  # [width, height]
            }
        except Exception as e:
            print(f"Warning: Could not extract camera calibration: {e}")
            # Return default calibration
            return {
                "camera_name": "CAM_FRONT",
                "intrinsic_matrix": [[1000, 0, 112], [0, 1000, 112], [0, 0, 1]],
                "translation": [0, 0, 0],
                "rotation": [1, 0, 0, 0],
                "sensor_token": None,
                "original_image_size": [1600, 900]  # Default size
            }


def test_file_saving():
    """Test function to verify file saving works correctly."""
    import tempfile
    import shutil
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"Testing file saving in: {temp_dir}")
    
    try:
        # Test with a small configuration
        extractor = NuScenesExtractor(
            nuscenes_root="/workspace/data/nuscenes/v1.0-mini",
            out_dir=temp_dir,
            split="train",
            input_frames=2,  # Reduced for testing
            input_interval=0.5,
            output_steps=3,  # Reduced for testing
            output_interval=0.5,
            img_size=(64, 64),  # Smaller images for testing
            steer_range=(-0.6, 0.6),
            accel_range=(-6.0, 6.0),
            wheelbase=2.7,
            jpg_quality=90,
            time_tolerance_ms=120,
        )
        
        # Extract a small number of samples
        extractor.extract_shards(max_samples_per_shard=10, verbose_every=1)
        
        # Check if output directory was created
        output_path = os.path.join(temp_dir, "train")
        if os.path.exists(output_path):
            files = os.listdir(output_path)
            print(f"✅ Output directory created successfully: {output_path}")
            print(f"✅ Generated {len(files)} shard files: {files}")
            return True
        else:
            print("❌ Output directory was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"🧹 Cleaned up temporary directory: {temp_dir}")


def main():
    """Main function to run the extractor."""
    dataroot = "/workspace/data/nuscenes/v1.0-mini"
    output_dir = "/workspace/data"
    
    extractor = NuScenesExtractor(
        nuscenes_root=dataroot,
        out_dir=output_dir,
        split="train",
        input_frames=4,
        input_interval=0.5,
        output_steps=10,
        output_interval=0.5,
        img_size=(224, 224),
        steer_range=(-0.6, 0.6),
        accel_range=(-6.0, 6.0),
        wheelbase=2.7,
        jpg_quality=90,
        time_tolerance_ms=120,
    )
    
    extractor.extract_shards(max_samples_per_shard=5000, verbose_every=1000)


if __name__ == "__main__":
    main()