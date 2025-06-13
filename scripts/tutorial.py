import os 
import tensorflow as tf
import math 
import numpy as np 
import itertools

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
     
# Set file path
file_path = '/mnt/dataset/e2e/training_202503292338.tfrecord-00000-of-00315'

# Load the dataset
dataset = tf.data.TFRecordDataset(file_path, compression_type='')

# Iterate over the dataset
for data in dataset:
    frame = open_dataset.DriverData()
    frame.ParseFromString(data.numpy())
    break


