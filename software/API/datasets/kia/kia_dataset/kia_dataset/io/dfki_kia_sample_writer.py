"""doc
# kia_dataset.io.dfki_kia_sample_writer

> Write samples in the kia format.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer
"""
import os
import json
import numpy as np
from typing import List

from kia_dataset.io.dfki_kia_dataset import KIADetection2D, KIADetection3D


class KIASampleWriter(object):
    def __init__(self, output: str, clear_output_folder: bool = True, use_object_id_as_key: bool = False):
        """
        Writes detection to a database.

        :param output_dir: The output directory where to write the results. The directory will be cleared before writing!
        :param clear_output_folder: If the output folder should be removed before writing. Default: True.
        :param use_object_id_as_key: If the key in the dictionary of annotations should be object id. When false, instance id as specified in E1.2.3 is used. Default: False.
        """
        self.output_dir = output
        self.use_object_id_as_key = use_object_id_as_key

        if clear_output_folder:
            if os.path.exists(self.output_dir):
                os.system("rm -rf {}".format(self.output_dir))
        os.makedirs(os.path.join(self.output_dir, "2d-bounding-box_json"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "3d-bounding-box_json"), exist_ok=True)

    def write(self, detections_2d: List[KIADetection2D], detections_3d: List[KIADetection3D], sample_token: str, world_to_ego=None, world_mode=True) -> None:
        """
        Write a list of samples for a frame.

        :param detections_2d: The 2d samples that should be written. (All must belong to the same frame and indices must align with 3d detections), when a value is None nothing is written (also no 3D).
        :param detections_3d: The 3d samples that should be written. (All must belong to the same frame and indices must align with 2d detections), when a value is None then only the 2D box is written.
        :param sample_token: The token for which the samples are (All must belong to the same frame).
        :param world_to_ego: Needs to be provided when detections are given in world coordinates.
        :param world_mode: Defines if the 3d box should be written only once in world coordinates. If false they are written in ego vehicle coordinates per camera. (Default: True).
        """
        frame_token = sample_token.split("/")[1]
        
        with open(os.path.join(self.output_dir, "2d-bounding-box_json/{}.json".format(frame_token)), "w") as f:
            boxes = {}
            for box_2d in detections_2d:
                box_2d_dict = box_2d._asdict()
                
                # Convert complex types to primitives
                for k in box_2d_dict:
                    if isinstance(box_2d_dict[k], np.ndarray):
                        box_2d_dict[k] = box_2d_dict[k].tolist()
                
                # Add primitive dict to datastructure.
                if self.use_object_id_as_key:
                    boxes[box_2d_dict["object_id"]] = box_2d_dict
                else:
                    boxes[box_2d_dict["instance_id"]] = box_2d_dict
            f.write(json.dumps(boxes, sort_keys=True, indent=2))
            f.flush()

        if world_mode:
            token_parts = sample_token.split("-")
            world_token = "world-" + token_parts[2] + "-" + token_parts[3] + "-" + token_parts[4]
        else:
            assert world_to_ego is not None
            world_token = frame_token
        with open(os.path.join(self.output_dir, "3d-bounding-box_json/{}.json".format(world_token)), "w") as f:
            boxes = {}
            for box_3d in detections_3d:
                box_3d_dict = box_3d._asdict()

                # If required transform to ego view.
                if not world_mode and world_to_ego is not None and box_3d is not None:
                    box_3d_dict["center"] = world_to_ego.apply_to_point(box_3d.center, single_point=True)
                    box_3d_dict["rotation"] = world_to_ego.q * box_3d.rotation

                # Convert complex types to primitives
                box_3d_dict["rotation"] = box_3d_dict["rotation"].elements
                for k in box_3d_dict:
                    if isinstance(box_3d_dict[k], np.ndarray):
                        box_3d_dict[k] = box_3d_dict[k].tolist()

                # Remove fields that make no sense
                if world_mode and "occlusion" in box_3d_dict:
                    del box_3d_dict["occlusion"]
                if "sensor" in box_3d_dict:
                    del box_3d_dict["sensor"]
                if "rotation" in box_3d_dict:
                    box_3d_dict["rot"] = box_3d_dict["rotation"]
                    del box_3d_dict["rotation"]

                # Add primitive dict to datastructure.
                if self.use_object_id_as_key:
                    boxes[box_3d_dict["object_id"]] = box_3d_dict
                else:
                    boxes[box_3d_dict["instance_id"]] = box_3d_dict
            f.write(json.dumps(boxes, sort_keys=True, indent=2))
            f.flush()

    def close(self) -> None:
        """
        Close the sample writer.
        """
        pass
