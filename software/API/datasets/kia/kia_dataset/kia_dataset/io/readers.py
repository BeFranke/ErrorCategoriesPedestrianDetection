"""doc
# kia_dataset.io.readers

> Readers for the data to make implementation of fixes and dataloaders easier.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer
* Philipp Heidenreich (Opel)
"""
import cv2
import json
import numpy as np
import os


def read_image_png(sequence_path, filename):
    """
    Read the png image as a numpy nd array.

    :param sequence_path: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
    :param filename: The filename of the actual frame without the ".png". This makes the name identical for all image based annotations.
    """
    path_filename_camera = os.path.join(sequence_path, 'sensor', 'camera', 'left', 'png', filename + '.png')
    return cv2.imread(path_filename_camera)


def read_instance_mask(sequence_path, filename):
    """
    Read the instance mask as a numpy nd array.
    
    The returned mask is of shape h,w and contains the integer values for the instance ids. Conversion from RGB to int is already done.

    :param sequence_path: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
    :param filename: The filename of the actual frame without the ".png"/".exr". This makes the name identical for all image based annotations.
    """
    # path_instance_exr = os.path.join(sequence_path, 'ground-truth', 'semantic-instance-segmentation_exr')
    path_instance_png = os.path.join(sequence_path, 'ground-truth', 'semantic-instance-segmentation_png')
    # if os.path.exists(path_instance_exr):   # tranche in [BIT_TRANCHE02]
    #     path_filename_instance_exr = os.path.join(path_instance_exr, filename + '.exr')
    #     instance = cv2.imread(path_filename_instance_exr, cv2.IMREAD_UNCHANGED)[:, :, 2].astype('int')
    if os.path.exists(path_instance_png): # tranche in [BIT_TRANCHE03, MAC_TRANCHE02]
        path_filename_instance_png = os.path.join(path_instance_png, filename + '.png')
        instance = cv2.imread(path_filename_instance_png).astype('int')
        instance = instance[:, :, 2] * 2**16 + instance[:, :, 1] * 2**8 + instance[:, :, 0]
    else:
        raise RuntimeError('Cannot find instance segmentation!')
    return instance


def read_depth(sequence_path, filename):
    """
    Read the depth map as a numpy nd array.
    
    The returned mask is of shape h,w and contains the float values for the depth in meters.

    :param sequence_path: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
    :param filename: The filename of the actual frame without the ".png"/".exr". This makes the name identical for all image based annotations.
    """
    path_depth_csv = os.path.join(sequence_path, 'ground-truth', 'depth_csv')
    path_depth_exr = os.path.join(sequence_path, 'ground-truth', 'depth_exr')
    if os.path.exists(path_depth_csv):      # tranche in [BIT_TRANCHE02]
        path_filename_depth_csv = os.path.join(path_depth_csv, filename + '.csv')
        depth = np.genfromtxt(path_filename_depth_csv, delimiter=',', usecols=range(1920), dtype='float')
    elif os.path.exists(path_depth_exr):    # tranche in [BIT_TRANCHE03, MAC_TRANCHE02]
        path_filename_depth_exr = os.path.join(path_depth_exr, filename + '.exr')
        depth = cv2.imread(path_filename_depth_exr, cv2.IMREAD_UNCHANGED)[:,:, 2]
    else:
        raise RuntimeError('Cannot find depth data.')
    return depth


def read_boxes_2d(sequence_path, filename):
    """
    Read the 2d bounding boxes as a dict mapping ids to boxes.
    
    The boxes returned are in the format the boxes have on the disk. No correction is done.

    :param sequence_path: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
    :param filename: The filename of the actual frame without the ".png"/".exr". This makes the name identical for all image based annotations.
    """
    with open(os.path.join(sequence_path, 'ground-truth', '2d-bounding-box_json', filename + '.json')) as f:
        boxes = json.load(f)
    return boxes


def read_boxes_3d(sequence_path, filename):
    """
    Read the 3d bounding boxes as a dict mapping ids to boxes.
    
    The boxes returned are in the format the boxes have on the disk. No correction is done.

    :param sequence_path: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
    :param filename: The filename of the actual frame without the ".png"/".exr". This makes the name identical for all image based annotations.
    """
    with open(os.path.join(sequence_path, 'ground-truth', '3d-bounding-box_json', filename + '.json')) as f:
        boxes = json.load(f)
    return boxes
