import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp
from PIL import Image
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils, plot_maps, geometry_utils

# TFRecord 파일 경로
FILEPATH = '/mnt/dataset/wod/perception_v1.4.3/archived_files/training/archived_files_training_training_0000/'
FILENAME = 'segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
WOMD_FILE = os.path.join(FILEPATH, FILENAME)
dataset = tf.data.TFRecordDataset(WOMD_FILE, compression_type='')

# 첫 번째 프레임 로딩
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytes(data.numpy()))
    if len(frame.laser_labels) > 0:
        print("Found frame with", len(frame.laser_labels), "labels")
        break
    break

def transform_point(p, pose):
    """Point [x, y, z] → vehicle local 좌표계로 변환"""
    global_xyz = np.array([p.x, p.y, p.z, 1.0])
    local = np.linalg.inv(np.reshape(pose, [4, 4])) @ global_xyz
    return local[0], local[1]  # x, y

# Load sample FRONT camera image and calibration
def extract_front_camera(frame):
    front_image = next(img for img in frame.images if img.name == open_dataset.CameraName.FRONT)
    calib = next(c for c in frame.context.camera_calibrations if c.name == open_dataset.CameraName.FRONT)
    return front_image, calib

def decode_image(image_proto):
    return tf.image.decode_jpeg(image_proto.image).numpy()

def compute_homography(calib, camera_pose, vehicle_pose):
    # Extract intrinsic matrix
    K = np.zeros((3, 3))
    K[0, 0] = calib.intrinsic[0]
    K[1, 1] = calib.intrinsic[1]
    K[0, 2] = calib.intrinsic[2]
    K[1, 2] = calib.intrinsic[3]
    K[2, 2] = 1.0

    # Camera to vehicle (extrinsic)
    cam_to_vehicle = np.array(calib.extrinsic.transform).reshape(4, 4)

    # Vehicle to world
    vehicle_to_world = np.array(vehicle_pose).reshape(4, 4)

    # Full camera pose in world
    cam_to_world = vehicle_to_world @ cam_to_vehicle

    # Define ground plane z=0 in world
    R = cam_to_world[:3, :3]
    t = cam_to_world[:3, 3]
    n = np.array([0, 0, 1])  # ground plane normal
    d = 0  # distance from origin

    H = K @ (R - (t[:, None] @ n[None, :]) / (d + n @ t))  # homography from ground to image
    return H / H[2, 2]

def project_bev_background(image_np, H, ax, size=100):
    from PIL import Image
    from skimage.transform import warp
    H_inv = np.linalg.inv(H)

    # Warp camera image onto ground
    output_shape = (size, size)
    warped = warp(image_np / 255.0, H_inv, output_shape=output_shape)

    ax.imshow(np.clip(warped, 0, 1), extent=[-50, 50, -50, 50], zorder=0)


def get_intrinsic_matrix(calib):
    K = np.zeros((3, 3))
    K[0, 0] = calib.intrinsic[0]  # fx
    K[1, 1] = calib.intrinsic[1]  # fy
    K[0, 2] = calib.intrinsic[2]  # cx
    K[1, 2] = calib.intrinsic[3]  # cy
    K[2, 2] = 1.0
    return K

def get_camera_pose(calib, vehicle_pose):
    cam_to_vehicle = np.array(calib.extrinsic.transform).reshape(4, 4)
    vehicle_to_world = np.array(vehicle_pose).reshape(4, 4)
    return vehicle_to_world @ cam_to_vehicle  # cam_to_world

def compute_ground_to_image_homography(camera_pose, K):
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    n = np.array([0, 0, 1])  # ground plane normal
    d = 0  # distance from origin

    H = K @ (R - np.outer(t, n) / (d + np.dot(n, t)))
    return H / H[2, 2]

# Store to return
image_np = None
H = None

# 차량의 포즈 정보 추출
vehicle_pose = np.array(frame.pose.transform).reshape(4, 4)

# 지도 정보 시각화
# print(frame)
# print(frame.map_features)
# print(frame.pose.transform)
fig, ax = plt.subplots(figsize=(8, 8))

for map_feature in frame.map_features:
    if map_feature.WhichOneof("feature_data") != "lane":
        continue

    polyline = map_feature.lane.polyline
    xs, ys = [], []
    for pt in polyline:
        x, y = transform_point(pt, frame.pose.transform)
        if abs(x) < 50 and abs(y) < 50:  # local 좌표계 기준 필터링
            xs.append(x)
            ys.append(y)

    if len(xs) > 1:
        ax.plot(xs, ys, color='blue', linewidth=1)
        
# 주변 차량 바운딩 박스 추가
for label in frame.laser_labels:
    print(f"Object at x={label.box.center_x:.2f}, y={label.box.center_y:.2f}")
    
    if abs(label.box.center_x) > 50 or abs(label.box.center_y) > 50:
        continue

    # 회전된 박스 좌표 계산
    corners = np.array([
        [ label.box.length/2,  label.box.width/2],
        [ label.box.length/2, -label.box.width/2],
        [-label.box.length/2, -label.box.width/2],
        [-label.box.length/2,  label.box.width/2],
        [ label.box.length/2,  label.box.width/2]
    ])
    rot = np.array([
        [np.cos(-label.box.heading), -np.sin(-label.box.heading)],
        [np.sin(-label.box.heading),  np.cos(-label.box.heading)]
    ])
    rotated = (rot @ corners.T).T + np.array([label.box.center_x, label.box.center_y])

    ax.plot(rotated[:,0], rotated[:,1], 'r-')

# 1. 이미지 및 캘리브레이션 불러오기
front_image = next(img for img in frame.images if img.name == open_dataset.CameraName.FRONT)
front_calib = next(c for c in frame.context.camera_calibrations if c.name == open_dataset.CameraName.FRONT)
img_np = tf.image.decode_jpeg(front_image.image).numpy()
K = get_intrinsic_matrix(front_calib)
cam_to_ego = np.array(front_calib.extrinsic.transform).reshape(4, 4)

# 2. Homography 계산
H = compute_ground_to_image_homography(cam_to_ego, K)
H_inv = np.linalg.inv(H)

print("Homography:\n", H)
print("Inverse:\n", H_inv)

# 3. BEV 배경에 이미지 투영
from skimage.transform import warp
img_norm = img_np / 255.0
bev_warp = warp(img_norm, H_inv, output_shape=(512, 512))  # 고해상도


plt.imshow(img_np)
plt.title("Original Camera Image")
plt.show()

plt.imshow(bev_warp)
plt.title("Warped BEV Image")
plt.show()

# 4. 시각화
ax.imshow(np.clip(bev_warp, 0, 1), extent=[-50, 50, -50, 50], zorder=0)

ax.set_title("BEV Map Features around ego")
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])
ax.set_aspect('equal')
plt.grid(True)
plt.show()





