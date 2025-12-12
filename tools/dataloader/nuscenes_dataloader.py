from gc import set_debug
import os 
import cv2 
import json 
import math 
import bisect 
import numpy as np
from typing import List, Dict, Tuple, Optional

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion
from nuscenes.can_bus.can_bus_api import NuScenesCanBus


CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

CATEGORIES = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
)

ATTRIBUTES = (
    'cycle.with_rider', 'cycle.without_rider',
    'pedestrian.moving', 'pedestrian.standing',
    'pedestrian.sitting_lying_down', 'vehicle.moving',
    'vehicle.parked', 'vehicle.stopped', 'None'
)

def get_nearest_message(timestamp: int, messages: List[Dict]) -> Dict:
    return min(messages, key=lambda x: abs(x['utime'] - timestamp))


if __name__ == "__main__":
    nusc_can = NuScenesCanBus(dataroot='./data/nuscenes')

    messages = nusc_can.get_messages('scene-0001', 'vehicle_monitor')
    nearest_message = get_nearest_message(1531883530949817, messages)
    breakpoint()


    nusc = NuScenes(version="v1.0-trainval", dataroot="./data/nuscenes", verbose=True)
    
    
    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])

    while True:
        data = {}
        sample = nusc.get("sample", sample['next']) if sample['token'] != scene['last_sample_token'] else None
        if sample is None:
            break
        
        # Get timestamp
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        data['timestamp'] = sd['timestamp']

        # Get ego pose
        sd = nusc.get('ego_pose', sd['ego_pose_token'])
        data['ego_pose'] = {
            'translation': sd['translation'],
            'rotation': sd['rotation'],
        }

        # Get camera images
        data['sensor'] = {}
        for cam in CAMERAS:
            img = nusc.get('sample_data', sample['data'][cam])
            calib = nusc.get('calibrated_sensor', img['calibrated_sensor_token'])
            data[cam] = {
                'intrinsic': calib['camera_intrinsic'],
                'translation': calib['translation'],
                'rotation': calib['rotation'],
                'img_size': [img['width'], img['height']],
                'img': cv2.imread(nusc.dataroot + '/' + img['filename'])
            }

        # Get object annotations
        data['object'] = {}
        for ann in sample['anns']:
            ann_data = nusc.get('sample_annotation', ann)
        breakpoint()

        # Get centerline annotations

        
    