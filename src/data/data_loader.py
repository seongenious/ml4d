from gc import set_debug
import os 
import cv2 
import json 
import math 
import bisect 
import numpy as np
from typing import List, Dict, Tuple, Optional

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion


class NuScenesExtractor:
    def __init__(self, nusc: NuScenes) -> None:
        self.nusc = nusc


def get_next_sample(sample: Dict) -> Dict:
    yield sample['next'] if 'next' in sample.keys() else None


CAM_LIST = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]


if __name__ == "__main__":
    nusc = NuScenes(version="v1.0-mini", dataroot="./data/nuscenes/v1.0-mini", verbose=False)
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
        for cam in CAM_LIST:
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

        
    