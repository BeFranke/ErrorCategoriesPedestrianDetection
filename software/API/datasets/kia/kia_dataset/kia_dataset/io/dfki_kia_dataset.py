"""doc
# kia_dataset.io.dfki_kia_dataset

> Prepare and read the KIA dataset.

## Preparation

It is assumed, that the zips/tars are downloaded into release and extracted to extracted. After extraction the autofix_data.py must be run on the data.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer
"""

import os
import cv2
import open3d
import json
import numpy as np
from typing import List
from collections import namedtuple
from pyquaternion import Quaternion
from deeptech.core.logging import warn
from deeptech.data import Dataset

from kia_dataset.io.annotations import cache
from kia_dataset.io.geometry import Transform, Projection
from kia_dataset.io.geometry.helpers import magnitude_of_vec, normalize_vec


CACHE_SIZE = 20
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLIT_TRAINVAL = "trainval"


_warned = {}
def warn_once(message: str):
    if message not in _warned:
        _warned[message] = True
        warn(message)


"""doc
## Definitions

* `SEMANTIC_MAPPING`: Is a dictionary from class name to color in the semantic annotation.
* `KIADetection2D`: Is a namedtuple for 2d detections. It extends a CommonDetection by velocity, truncated, instance_id and 
* `KIADetection3D`: Is a namedtuple for 3d detections. It extends a CommonDetection by velocity, instance_id and object_id.
"""
KIADetection2D = namedtuple("KIADetection2D", ["class_id", "sensor", "center", "size", "rotation", "confidence", "occlusion", "velocity", "truncated", "instance_id", "object_id"])
KIADetection3D = namedtuple("KIADetection3D", ["class_id", "sensor", "center", "size", "rotation", "confidence", "occlusion", "velocity", "instance_id", "object_id"])
SEMANTIC_MAPPING = {
    "unlabeled": (0, 0, 0),
    
    "animal": (100, 90, 0),
    "construction_worker": (220, 20, 200),
    "person": (220, 20, 60),
    "walk_assistance": (220, 20, 100),
    "child": (220, 20, 0),
    "kinderwagen": (220, 20, 175),
    "police_officer": (220, 20, 225),
    "rollstuhlfahrer": (220, 20, 150),
    "cyclist": (255, 64, 64),
    "rider": (255, 0, 0),
    
    "car": (0, 0, 142),
    "trailer": (0,0,110),
    "construction_vehicle": (0,0,80),
    "bus": (0, 60, 100),
    "bicycle": (119, 11, 32),
    "truck": (0,0, 70),
    "motorcycle": (0, 0, 230),
    "police_car": (0, 0, 155),
    "van": (0, 0, 142),
    
    "dynamic": (111, 74, 0),
    
    "parking": (250, 170, 160),
    "pole": (153, 153, 153),
    "trafic_light": (250, 170, 30),
    "traffic_sign": (220, 220, 0),
    
    "ground": (81, 0, 81),
    "lane_marking_bit": (255, 255, 0),
    "lane_marking_mv": (255, 255, 255),
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "rail track": (230, 150, 140),
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "guard rail": (180, 165, 180),
    "bridge": (150, 100, 100),
    "tunnel": (150, 120, 90),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "sky": (70, 130, 180),
    "caravan": (0, 0, 90),
    "train": (0, 80, 100),
    "license plate": (0, 0, 142)
}


"""doc
# Using the KIA-Dataset
"""
class KIADataset(Dataset):
    def __init__(self,
    split: str,
    InputType,
    OutputType,
    data_path: str,
    data_sequences: List[str],
    data_use_world_coordinates: bool = True,
    data_exr_images: bool = False,
    ):
        """
        The KIA Dataset is a synthetic dataset containing annotations for various tasks in automated driving.

        You can use the dataset and directly access elements or use it as a pytorch dataloader during training.
        ```
        InputType = NamedTuple("InputType", image=np.ndarray, scan=np.ndarray, projection=Projection)
        OutputType = NamedTuple("OutputType", detections_3d=List[KIADetection3D], sample_token=str)
        
        dataset = KIADataset(split, InputType, OutputType, data_path)
        inp: InputType, outp: OutputType = dataset[42]

        dataset.transformers.append(MyTransformer())

        dataloader = dataset.to_pytorch(batch_size)
        for inp, outp in dataloader:
            preds = net(*inp)
            loss = myLoss(preds, outp)
            ...
        ```

        :param split: Specifies for what the data is used train/val/test.
        :param InputType: The type definition (namedtuple) that the network input (features) should have.
        :param OutputType: The tye definition (namedtuple) that the network output (labels) should have.
        :param data_path: "/fakepath/KIA_dataset"` The path where the dataset is located.
        :param data_sequences: A list of sequences to use (e.g. kia_dataset.split.TRAIN_RELEASE_2).
        :param data_use_world_coordinates: If 3D annotations should be in world coordinates, when set to False the ego vehicle coordinate system will be used. (Default: True).
        :param data_exr_images: Controls if to use the exr image format for the RGB images. (Default: False)
        """
        super().__init__(split, InputType, OutputType)
        self.use_world_coordinates = data_use_world_coordinates
        self.data_path = data_path
        self.exr_image = data_exr_images
        self.all_sample_tokens = self.get_all_sample_tokens(data_sequences)

    def _get_version(self) -> str:
        return "KIA_v1"
    
    def get_all_sample_tokens(self, data_sequences: List[str]) -> List[str]:
        """
        A function that computes the sample tokens assigned to a sequence and company.
        
        :param data_sequences: A list of sequences that are allowed.
        :return: A list of sample tokens that fulfill the requirements.
        """
        def _get_files(folder):
            fileSet = set() 

            for root, dirs, files in os.walk(folder):
                for fileName in files:
                    full_path = os.path.join(root[len(folder)+1:], fileName)
                    fileSet.add(full_path)
            return list(fileSet)

        frames = [f for f in _get_files(self.data_path) if "sensor/camera/left/png/" in f]
        sample_tokens = []
        for f in frames:
            sample_token = f.split("/")[-1].replace(".png", "")
            sequence = f.split("/")[0]
            if sequence in data_sequences:
                company = sequence.split("_")[0]
                sample_tokens.append(f"{company}/{sample_token}")
        return sorted(sample_tokens)

    @cache(CACHE_SIZE)
    def get_sample_token(self, sample_token: str) -> str:
        return sample_token

    @cache(CACHE_SIZE)
    def get_sensor(self, sample_token: str) -> str:
        return sample_token.split("-")[0].split("/")[-1] + "-" + sample_token.split("-")[1]

    @cache(CACHE_SIZE)
    def get_sequence(self, sample_token: str) -> str:
        return sample_token.split("/")[0] + "_results_sequence_" + sample_token.split("-")[2] + "-" + sample_token.split("-")[3]

    @cache(CACHE_SIZE)
    def get_frame(self, sample_token: str) -> str:
        return sample_token.split("/")[1]

    @cache(CACHE_SIZE)
    def get_world_token(self, sample_token: str) -> str:
        token_parts = sample_token.split("-")
        world_token = "world-" + token_parts[2] + "-" + token_parts[3] + "-" + token_parts[4]
        return world_token

    @cache(CACHE_SIZE)
    def get_image(self, sample_token: str) -> np.ndarray:
        """
        Load the image from disk.

        Loads the exr if self.exr_image is true, else it will load the png.

        :param sample_token: The sample token for which to load the image.
        :return: The image as a numpy array.
        """
        if self.exr_image:
            fname = "{}/{}/sensor/camera/left/exr/{}.exr".format(self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token))
            rgb_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)[...,::-1]
        else:
            fname = "{}/{}/sensor/camera/left/png/{}.png".format(self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token))
            rgb_img = cv2.imread(fname)[...,::-1]
        return rgb_img
    
    @cache(CACHE_SIZE)
    def get_depth(self, sample_token: str) -> np.ndarray:
        """
        Load the depth data from disk.

        :param sample_token: The sample token for which to load the depth.
        :return: The depth map as a numpy array.
        """
        fname = "{}/{}/ground-truth/depth_exr/{}.exr".format(self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token))
        if os.path.exists(fname):
            # E1.2.3 official format
            depth = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

            # We found some cases where depth was stored in channel 3...
            if depth.shape[2] == 3:
                depth = depth[:, :, 2]
        else:
            # BIT Tranche 2
            fname = "{}/{}/ground-truth/depth_png/{}.png".format(self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token))
            depth = cv2.imread(fname)
        return depth

    @cache(CACHE_SIZE)
    def get_scan(self, sample_token: str) -> np.ndarray:
        """
        Load the lidar scan data from disk.

        :param sample_token: The sample token for which to load the depth.
        :return: The lidar scan poincloud as a numpy array. Shape is N,6 where N is the number of points, 0-3 is xyz and 4-6 is color info.
            (likely to change in the future)
        """
        fname = "{}/{}/sensor/lidar/pcd/{}.pcd".format(self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token).replace("camera", "lidar"))
        if os.path.exists(fname):
            pcd_o3d = open3d.io.read_point_cloud(fname)
            pcd = np.asarray(pcd_o3d.points)
            colors = np.asarray(pcd_o3d.colors)

            transform = self._get_scan_to_world(sample_token)
            if not self.use_world_coordinates:
                transform = transform.then(self.get_world_to_ego(sample_token))

            pcd[:, :3] = transform.apply_to_point(pcd[:, :3].T, single_point=False).T
        else:
            warn_once("No LiDAR in data. Fake scan with 0 points.")
            pcd = np.zeros((0, 3), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.float32)
        return np.concatenate([pcd, colors], axis=1)

    @cache(CACHE_SIZE)
    def get_semantic_segmentation(self, sample_token: str) -> np.ndarray:
        """
        Load the semantic segmentation data from disk.

        :param sample_token: The sample token for which to load the semantic segmentation.
        :return: The semantic segmentation is loaded as an image in np ndarray.
        """
        fname = "{}/{}/ground-truth/semantic-group-segmentation_png/{}.png".format(self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token))
        return cv2.imread(fname)[...,::-1]

    @cache(CACHE_SIZE)
    def get_instance_segmentation(self, sample_token: str) -> np.ndarray:
        """
        Load the instance segmentation data from disk.

        :param sample_token: The sample token for which to load the instance segmentation.
        :return: The instance segmentation is loaded as an image in np ndarray.
        """
        fname = "{}/{}/ground-truth/semantic-instance-segmentation_png/{}.png".format(self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token))
        if os.path.exists(fname):
            # E1.2.3 official format
            img_instance = cv2.imread(fname)
        else:
            # Legacy format from early releases
            fname = "{}/{}/ground-truth/semantic-instance-segmentation_exr/{}.exr".format(self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token))
            img_instance = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            img_instance = img_instance[:,:,2].astype('uint16')
        return img_instance

    @cache(CACHE_SIZE)
    def get_detections_2d(self, sample_token):
        # Try loading the hotfix first, else load the correct data
        fname = "{}/{}/ground-truth/2d-bounding-box-filtered-occlusion_json/{}.json".format(self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token))
        if not os.path.exists(fname):
            # E1.2.3 official filename
            fname = "{}/{}/ground-truth/2d-bounding-box_json/{}.json".format(self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token))
        with open(fname, "r") as f:
            data = json.loads(f.read())
            
        detections_2d = []
        for k in data:
            val = data[k]

            # Confidence only exists for predictions therfore set it to 1 for gt where it is missing intentionally.
            confidence = val["confidence"] if "confidence" in val else 1.0

            # Fill all non existing fields with default values.
            occlusion = val["occlusion"] if "occlusion" in val else -1
            truncated = val["truncated"] if "truncated" in val else False
            if "center" not in val:  # Official E1.2.3 (V3.0 mode)
                center = [val["c_x"], val["c_y"]] if "c_x" in val and "c_y" in val else [np.nan, np.nan]
                size = [val["w"], val["h"]] if "w" in val and "h" in val else [np.nan, np.nan]
                velocity = [val["v_x"], val["v_y"]] if "v_x" in val and "v_y" in val else [np.nan, np.nan]
            else:  # DFKI KIASampleWriter format
                center = val["center"] if "center" in val else [np.nan, np.nan]
                size = val["size"] if "size" in val else [np.nan, np.nan]
                velocity = val["velocity"] if "velocity" in val else [np.nan, np.nan]
            instance_id = val["instance_id"] if "instance_id" in val else k
            object_id = val["object_id"] if "object_id" in val else k

            class_id = "Unknown"
            if "class" in val:
                class_id = val["class"]
            if "category" in val:
                class_id = val["category"]

            detection = KIADetection2D(
                class_id=class_id,
                sensor=self.get_sensor(sample_token),
                center=np.array(center),
                size=np.array(size),
                rotation=0,
                confidence=confidence,
                occlusion=occlusion,
                velocity=np.array(velocity),
                truncated=truncated,
                instance_id=instance_id,
                object_id=object_id
            )
            detections_2d.append(detection)
        return detections_2d

    @cache(CACHE_SIZE)
    def convert_legacy_3d_box_format(self, box):
        x1 = np.array(box["x1"])
        x2 = np.array(box["x2"])
        x3 = np.array(box["x3"])
        x4 = np.array(box["x4"])
        x5 = np.array(box["x5"])
        x6 = np.array(box["x6"])
        x7 = np.array(box["x7"])
        x8 = np.array(box["x8"])
        
        # Compute scale dependent direction vectors
        FrontVec = x1-x5  # Vector defining the edge towards the front
        UpVec = x2-x1  # Vector defining the edge towards the top
        LeftVec = x3-x1  # Vector defining the edge towards the left

        # Use scale dependent direction vectors
        Length = magnitude_of_vec(FrontVec)
        Width = magnitude_of_vec(LeftVec)
        Height = magnitude_of_vec(UpVec)
        Center = np.array([x1,x2,x3,x4,x5,x6,x7,x8]).mean(axis=0)

        # Normalize direction vectors (required for change of basis)
        #FrontVec = np.array(normalize_vec(FrontVec))
        #LeftVec = np.array(normalize_vec(LeftVec))
        #UpVec = np.array(normalize_vec(UpVec))
        # FIXME: Currently, assume Up to be z axis and then construct orientation from front vector.
        FrontVec = np.array([FrontVec[0], FrontVec[1], 0])
        FrontVec = np.array(normalize_vec(FrontVec))
        LeftVec = np.array([-FrontVec[1], FrontVec[0], 0])
        UpVec = np.array([0,0,1])

        # Rotation anno2world is an easy change of basis (assuming orthogonal directions)
        MatrixFromColumns = lambda a,b,c: np.stack((a,b,c), axis=-1)
        R_anno2world = MatrixFromColumns(FrontVec, LeftVec, UpVec)

        # Leading to
        box["size"] = [Length, Width, Height]
        box["center"] = Center  # xyz
        box["rot"] = Quaternion(matrix=R_anno2world).elements  # convert rotation matrix to quaternion elements
        return box

    @cache(CACHE_SIZE)
    def get_detections_3d(self, sample_token):
        world_token = self.get_world_token(sample_token)
        world_to_ego: Transform = self.get_world_to_ego(sample_token)
        world_to_ego_rotation = world_to_ego.unsafe_q
        fname = "{}/{}/ground-truth/3d-bounding-box_json/{}.json".format(self.data_path, self.get_sequence(sample_token), world_token)
        if not os.path.exists(fname):
            fname = fname.replace("world-", "world_")
            warn_once("Using fallback prefix 'world_', as 'world-' did not exist.")

        with open(fname, "r") as f:
            data = json.loads(f.read())

        detections_3d = []
        for k in data:
            val = data[k]
            if "x1" in val: # Convert legacy format
                val = self.convert_legacy_3d_box_format(val)

            # Confidence only exists for predictions therfore set it to 1 for gt where it is missing intentionally.
            confidence = val["confidence"] if "confidence" in val else 1.0

            # Cautiously load the data...
            instance_id = val["instance_id"] if "instance_id" in val else k
            center = val["center"] if "center" in val else [np.nan, np.nan, np.nan]
            size = val["size"] if "size" in val else [np.nan, np.nan, np.nan]
            velocity = val["velocity"] if "velocity" in val else [np.nan, np.nan, np.nan]
            rotation = val["rot"] if "rot" in val else [np.nan, np.nan, np.nan, np.nan]
            object_id = val["object_id"] if "object_id" in val else k
            class_id = "Unknown"
            if "class_id" in val:
                class_id = val["class_id"]
            if "class" in val:
                class_id = val["class"]
            if "category" in val:
                class_id = val["category"]
            
            detection = KIADetection3D(
                class_id=class_id,
                sensor=self.get_sensor(sample_token),
                center=np.array(center),
                size=np.array(size),
                rotation=Quaternion(*rotation),  # w x y z
                confidence=confidence,
                occlusion=np.nan,
                velocity=np.array(velocity),
                instance_id=instance_id,
                object_id=object_id
            )
            if not self.use_world_coordinates:
                new_center = world_to_ego.apply_to_point(detection.center)
                new_velocity = world_to_ego.apply_to_direction(detection.velocity)
                new_rotation = detection.rotation * world_to_ego_rotation  # FIXME unsure on order
                detection = detection._replace(center=new_center, rotation=new_rotation, velocity=new_velocity)
            detections_3d.append(detection)

        return detections_3d

    @cache(CACHE_SIZE)
    def get_skeletons_2d(self, sample_token):
        raise NotImplementedError()

    @cache(CACHE_SIZE)
    def get_skeletons_3d(self, sample_token):
        raise NotImplementedError()

    @cache(CACHE_SIZE)
    def get_projection(self, sample_token) -> Projection:
        fname = "{}/{}/ground-truth/matrices_csv/{}_matrix_P.csv".format(self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token))
        if os.path.exists(fname):
            # BIT Tranche 2 format
            with open(fname, "r") as f:
                data = f.read()
            lines = data.split("\n")
            arr = []
            for line in lines:
                if line != "":
                    line = line.split(",")
                    arr.append([float(x) for x in line if x != ""])
            P_world_to_cam = np.array(arr)
            P = np.dot(P_world_to_cam, self._get_cam_to_world(sample_token).Rt)
        else:
            raise NotImplementedError()

        projection = Projection(P=P)
        if self.use_world_coordinates:
            projection = projection.after(self._get_cam_to_world(sample_token).inverse)
        else:
            projection = projection.after(self._get_cam_to_ego(sample_token))

        return projection

    @cache(CACHE_SIZE)
    def _get_scan_to_world(self, sample_token) -> Transform:
        fname = "{}/{}/ground-truth/matrices_csv/{}-matrix-lidar-rel.csv".format(
            self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token).replace("camera", "lidar"))
        if os.path.exists(fname):
            # BIT Tranche 3 format
            with open(fname, "r") as f:
                data = f.read()
            lines = data.split("\n")
            arr = []
            for line in lines:
                if line != "":
                    line = line.split(",")
                    arr.append([float(x) for x in line if x != ""])
            scan_to_world = Transform(Rt=np.array(arr))
        else:
            raise NotImplementedError()

        return scan_to_world

    @cache(CACHE_SIZE)
    def _get_cam_to_world(self, sample_token) -> Transform:
        fname = "{}/{}/ground-truth/matrices_csv/{}_matrix_abs.csv".format(
            self.data_path, self.get_sequence(sample_token), self.get_frame(sample_token))
        if os.path.exists(fname):
            # BIT Tranche 4 Format
            with open(fname, "r") as f:
                data = f.read()
            lines = data.split("\n")
            arr = []
            for line in lines:
                if line != "":
                    line = line.split(",")
                    arr.append([float(x) for x in line if x != ""])
            cam_to_world = Transform(Rt=np.array(arr))
        else:
            raise NotImplementedError()

        return cam_to_world.inverse

    @cache(CACHE_SIZE)
    def _get_cam_to_ego(self, sample_token) -> Transform:
        return Transform(Rt=np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ]))

    @cache(CACHE_SIZE)
    def get_world_to_ego(self, sample_token) -> Transform:
        world_to_cam = self._get_cam_to_world(sample_token).inverse
        cam_to_ego=self._get_cam_to_ego(sample_token)
        return world_to_cam.then(cam_to_ego.inverse)

    @cache(CACHE_SIZE)
    def get_ego_to_world(self, sample_token) -> Transform:
        return self.get_world_to_ego(sample_token).inverse
