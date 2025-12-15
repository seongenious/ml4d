from gc import set_debug
import os 
import cv2 
import json 
import math 
import bisect 
import numpy as np
from typing import Any, List, Dict, Tuple, Optional
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion
from nuscenes.can_bus.can_bus_api import NuScenesCanBus


CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

NUSCENES_CATEGORIES = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
)

NUSCENES_ATTRIBUTES = (
    'cycle.with_rider', 'cycle.without_rider',
    'pedestrian.moving', 'pedestrian.standing',
    'pedestrian.sitting_lying_down', 'vehicle.moving',
    'vehicle.parked', 'vehicle.stopped', 'None'
)

NUSCENES_NAME_MAPPING = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.stroller': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'    
}


class NuScenesDataloader:
    def __init__(self, dataroot: str, version: str) -> None:
        # Initialize nuScenes and can bus
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.nusc_can = NuScenesCanBus(dataroot=dataroot)

        # Split scenes by version
        if version == 'v1.0-trainval':
            train_scenes = set(splits.train)
            val_scenes = set(splits.val)
        elif version == 'v1.0-test':
            train_scenes = set(splits.test)
            val_scenes = []
        elif version == 'v1.0-mini':
            train_scenes = set(splits.mini_train)
            val_scenes = set(splits.mini_val)
        else:
            raise ValueError(f'Invalid version: {version}')

        # Convert scene names to tokens
        train_scenes = set(
            [s['token'] for s in self.nusc.scene if s['name'] in train_scenes])
        val_scenes = set(
            [s['token'] for s in self.nusc.scene if s['name'] in val_scenes])

        # Extract train and val scenes
        train_infos = []
        for scene_token in tqdm(train_scenes, desc='Extracting train scenes'):
            scene = self.nusc.get('scene', scene_token)
            samples = self._extract_scene(scene)
            train_infos.append(samples)
        self.train_scenes = train_infos

        val_infos = []
        for scene_token in tqdm(val_scenes, desc='Extracting val scenes'):
            scene = self.nusc.get('scene', scene_token)
            samples = self._extract_scene(scene)
            val_infos.append(samples)
        self.val_scenes = val_infos

    def _extract_scene(self, scene: Any) -> List[Dict]:
        samples = []
        sample = self.nusc.get('sample', scene['first_sample_token'])
        while sample is not None:
            data = self._extract_sample(sample)
            samples.append(data)
            if sample['token'] != scene['last_sample_token']:
                sample = self.nusc.get('sample', sample['next'])
            else:
                break
        return samples

    def _extract_sample(self, sample: Any) -> Dict:
        data = {}

        # Get lidar data for annotation transform to global frame
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_rec = self.nusc.get('sample_data', lidar_token)
        ego_rec = self.nusc.get('ego_pose', lidar_rec['ego_pose_token'])
        cs_rec = self.nusc.get(
            'calibrated_sensor', lidar_rec['calibrated_sensor_token'])
        
        _, boxes, _ = self.nusc.get_sample_data(lidar_token)
        lidar2ego_t = cs_rec['translation']
        lidar2ego_r = cs_rec['rotation']
        lidar2ego_r_mat = Quaternion(lidar2ego_r).rotation_matrix
        ego2global_t = ego_rec['translation']
        ego2global_r = ego_rec['rotation']
        ego2global_r_mat = Quaternion(ego2global_r).rotation_matrix
        
        data['timestamp'] = lidar_rec['timestamp']
        data['translation'] = ego_rec['translation']
        data['rotation'] = ego_rec['rotation']

        # Get camera images
        data['sensor'] = {}
        for cam in CAMERAS:
            cam_token = sample['data'][cam]
            cam_rec = self.nusc.get('sample_data', cam_token)
            calib_rec = self.nusc.get(
                'calibrated_sensor', cam_rec['calibrated_sensor_token'])
            data['sensor'][cam] = {
                'intrinsic': calib_rec['camera_intrinsic'],
                'translation': calib_rec['translation'],
                'rotation': calib_rec['rotation'],
                'img_size': [cam_rec['width'], cam_rec['height']],
                'img_path': os.path.join(
                    self.nusc.dataroot, cam_rec['filename'])
            }

        # Get object annotations
        data['label'] = {}
        annotations = [
            self.nusc.get('sample_annotation', token) for token in sample['anns']
        ]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array(
            [b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        corners = np.array([b.corners() for b in boxes])
        if corners.ndim == 3:
            corners = corners.transpose(0, 2, 1)
        else:
            corners = (corners.T)[None, ...]
        
        velocity = np.array(
            [self.nusc.box_velocity(token)[:2] for token in sample['anns']])
        for i in range(len(velocity)):
            vel_3d = np.array([*velocity[i], 0.0])
            vel_3d = vel_3d @ np.linalg.inv(ego2global_r_mat).T  # global to ego
            velocity[i] = vel_3d[:2]

        valid = np.array(
            [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0 
             for anno in annotations],
            dtype=bool).reshape(-1)

        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in NUSCENES_NAME_MAPPING:
                names[i] = NUSCENES_NAME_MAPPING[names[i]]
        names = np.array(names)

        data['label']['object'] = {
            'reg': np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1),
            'cls': names,
            'vel': velocity,
            'valid': valid,
            'corners': corners,
        }

        return data

    def get_train_scenes(self) -> List[Dict]:
        return self.train_scenes

    def get_val_scenes(self) -> List[Dict]:
        return self.val_scenes

    def get_nearest_message(self, timestamp: int, messages: List[Dict]) -> Dict:
        return min(messages, key=lambda x: abs(x['utime'] - timestamp))

if __name__ == "__main__":
    # nusc_can = NuScenesCanBus(dataroot='./data/nuscenes')

    # messages = nusc_can.get_messages('scene-0001', 'vehicle_monitor')
    # nearest_message = get_nearest_message(1531883530949817, messages)
    # breakpoint()

    dataloader = NuScenesDataloader(dataroot='./data/nuscenes', version='v1.0-trainval')

        
    