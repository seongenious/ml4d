from gc import set_debug
import os 
import cv2 
import json 
import numpy as np
from typing import Any, List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from nuscenes_dataloader import NuScenesDataloader
from utils import get_2d_corners

CAM_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
]

CLASS_COLORS = {
    'car': (0.8, 0, 0.1),
    'truck': (0.9, 0, 0.3),
    'trailer': (0.9, 0, 0.5),
    'bus': (0.9, 0, 0.7),
    'construction_vehicle': (0.5, 0.5, 0.5),
    'bicycle': (1.0, 0.1, 0),
    'motorcycle': (1.0, 0.2, 0),
    'pedestrian': (0.7, 0.7, 0.7),
    'human.pedestrian.stroller': (0.7, 0.7, 0.7),
    'traffic_cone': (0, 0, 0),
    'barrier': (0.2, 0.2, 0.2),
}

def visualize_sample(sample: Dict) -> None:
    # Set figure size and grid spec
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 2])

    for i, cam in enumerate(CAM_ORDER):
        # Get image
        cam_data = sample['sensor'][cam]
        img = cv2.imread(cam_data['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Add image to figure
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(cam)

    # Add BEV image
    ax_bev = fig.add_subplot(gs[:, 3])
    visualize_bev(sample, ax_bev)

    plt.tight_layout()
    plt.show()

def visualize_bev(sample: Dict, ax: plt.Axes) -> None:
    reg = np.array(sample['label']['object']['reg'])
    cls = np.array(sample['label']['object']['cls'])
    vel = np.array(sample['label']['object']['vel'])
    valid = np.array(sample['label']['object']['valid'])
    corners3d = np.array(sample['label']['object']['corners'])
    corners = get_2d_corners(reg[:, :3], reg[:, 3:6], reg[:, 6:])
    

    for i in range(len(reg)):
        if valid[i]:
            corner = corners[i]
            corner = np.vstack([corner, corner[0]])
            ax.plot(corner[:, 0], corner[:, 1], color=CLASS_COLORS[cls[i]])
            ax.scatter(reg[:, 0], reg[:, 1])

    ax.set_title('Bird-Eye-View')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim((-50, 50))
    ax.set_ylim((-50, 100))
    # ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)    

if __name__ == "__main__":
    if os.path.exists('sample.json'):
        sample = json.load(open('sample.json', 'r'))
    else:
        dataloader = NuScenesDataloader(dataroot="./data/nuscenes", version="v1.0-trainval")
        scene = dataloader.get_train_scenes()[0]
        sample = scene[0]

        def to_serializable(obj):
          if isinstance(obj, np.ndarray):
              return obj.tolist()
          if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
              return None
          return obj

        json.dump(sample, open('sample.json', 'w'), indent=4, default=to_serializable)

    visualize_sample(sample)