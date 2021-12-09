from typing import List
import cv2
import numpy as np

from dtk.visualizations.detections import plot_2d, plot_bev, plot_3d, colorize_distances

from kia_dataset.io.dfki_kia_dataset import KIADetection2D, KIADetection3D, SEMANTIC_MAPPING
from kia_dataset.io.geometry import Projection


def _caption_image(sample_token, img, title):
    statistics = title
    statistics += " | "
    statistics += "dtype={}, ".format(img.dtype)
    statistics += "shape={}, ".format(img.shape)
    statistics += "min={}, ".format(img.min())
    statistics += "max={}, ".format(img.max())
    statistics += "avg={:.1f}, ".format(img.mean())
    statistics += "std={:.1f}, ".format(img.std())
    statistics += "unique={:.1f}".format(np.unique(img).size)
    if len(img.shape) == 2:
        img = np.array(img)
        img = (colorize_distances(img) * 255).astype(np.uint8)
    img = np.array(img[:,:,::-1])
    cv2.putText(img=img, text=statistics, org=(2,17),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 255, 255))
    cv2.putText(img=img, text=sample_token, org=(2,img.shape[0]-5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,255,255))
    return img[:,:,::-1]


def visualize_depth(depth: np.ndarray, sample_token: str, max_dist: float = 100.0) -> np.ndarray:
    depth[depth==depth.max()] = 0
    depth[depth > max_dist] = max_dist
    return _caption_image(sample_token, depth, "depth")


def visualize_semantic_segmentation(semantic_segmentation: np.ndarray, sample_token: str) -> np.ndarray:
    return _caption_image(sample_token, semantic_segmentation, "semantic_segmentation")


def visualize_semantic_segmentation_mislabeled(semantic_segmentation: np.ndarray, sample_token: str) -> np.ndarray:
    semantics_error = semantic_segmentation.copy()
    for allowed_key in SEMANTIC_MAPPING.values():
            r1, g1, b1 = allowed_key
            r2, g2, b2 = 0, 0, 0
            red, green, blue = semantics_error[:,:,0], semantics_error[:,:,1], semantics_error[:,:,2]
            mask = (red == r1) & (green == g1) & (blue == b1)
            semantics_error[:,:,:3][mask] = [r2, g2, b2]
    return _caption_image(sample_token, semantic_segmentation, "semantic_segmentation_error")


def visualize_instance_segmentation(instance_segmentation: np.ndarray, sample_token: str) -> np.ndarray:
    return _caption_image(sample_token, instance_segmentation, "instance_segmentation")


def visualize_box2d(image: np.ndarray, detections_2d: List[KIADetection2D], sample_token: str) -> np.ndarray:
    img = plot_2d(detections_2d, image)
    return _caption_image(sample_token, img, "box2d")


def visualize_box3d(image: np.ndarray, detections_3d: List[KIADetection3D], projection: Projection, sample_token: str) -> np.ndarray:
    img = plot_2d(detections_3d, image, projection)
    return _caption_image(sample_token, img, "box3d")


def visualize_scan_depth(image: np.ndarray, scan: np.ndarray, detections_3d: List[KIADetection3D], projection: Projection, sample_token: str):
    img = projection.pointcloud_to_depth_image(scan=scan[:, :3].T, reference_image_shape=image.shape)[:, :, 0]
    img[img > 100.0] = 100.0
    if len(img.shape) == 2:
        img = (colorize_distances(img) * 255).astype(np.uint8)
    img = plot_2d(detections_3d, img, projection)
    return _caption_image(sample_token, img, "box3d_depth")


def visualize_scan_bev(scan: np.ndarray, detections_3d: List[KIADetection3D], sample_token: str, ego_to_world=None):
    if ego_to_world is not None:
        ego_position = ego_to_world.t
    else:
        ego_position = np.array([0.0, 0.0, 0.0])
    img = plot_bev(detections_3d, scan[:, :3], ego_position=ego_position)
    return _caption_image(sample_token, img, "box3d_bev")


def visualize_scan_3d(scan: np.ndarray, detections_3d: List[KIADetection3D], ego_to_world=None):
    if ego_to_world is not None:
        ego_position = ego_to_world.t
    else:
        ego_position = np.array([0.0, 0.0, 0.0])
    filtered_detections = []
    for detection in detections_3d:
        dist2 = ((detection.center - ego_position) ** 2).sum()
        if dist2 < 50 * 50:
            filtered_detections.append(detection)
    return plot_3d(filtered_detections, scan[:, :3], colors=scan[:, 3:], ego_position=ego_position)
