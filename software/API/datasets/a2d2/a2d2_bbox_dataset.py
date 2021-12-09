import json
import os
from typing import Tuple

from PIL import Image
import numpy as np

from datasets.base_dataset import BaseDataset


class A2D2BboxDataset(BaseDataset):
    """
    This class represents the data for an Audi dataset. The class inherits
    from the 'Dataset' class and is compatible to all the training, inference
    and evaluation scripts.

    This class splits the dataset along the different sequences. If there are
    ten sequences you might want to use 8 for training, 1 for validation and
    1 for testing. You need to specify that split in the splits folder. Please
    find further instructions for the structure of the dataset on disk in the
    README_2.md file.
    """

    def __init__(self, config, image_set, mode, augmentation) -> None:
        super().__init__(config, image_set,  mode, augmentation)
        self._remove_bad_images()

    def _get_image_set(self):
        return self._image_set

    def _get_dataset_root_path(self) -> str:
        return self._config.a2d2_root

    def _load_image_ids(self) -> list:
        sequences, ids = [], []
        images_root_path = os.path.join(self._root_path, self._image_set)

        splits = self._get_sequence_splits()

        for seq in splits[self._mode]:
            sequence_path = os.path.join(images_root_path, seq,
                                         "camera/cam_front_center/")
            files = os.listdir(sequence_path)
            ids += [seq + "/" + name[:-4] for name in files if
                    name.endswith('.png')]
        return ids

    def _get_class_names(self) -> list:
        return ["background", 'Animal', 'Bicycle', 'Bus', 'Car',
                'Caravan Transporter', 'Cyclist', 'Emergency Vehicle',
                'Motor Biker', 'Motorcycle', 'Pedestrian', 'Trailer', 'Truck',
                'Utility Vehicle', 'VanSUV']

    def _load_image_sizes(self) -> Tuple[list, list]:
        img_widths, img_heights = [], []
        for path in self._image_paths:
            image = Image.open(path, mode='r')
            img_widths.append(image.width)
            img_heights.append(image.height)
        return img_widths, img_heights

    def _load_image_paths(self) -> list:
        image_paths = []
        images_root_path = os.path.join(self._root_path, self._image_set)
        for single_id in self.image_ids:
            seq = single_id.split("/")[0]
            file = single_id.split("/")[1] + ".png"
            path = os.path.join(images_root_path, seq,
                                "camera/cam_front_center/", file)
            image_paths.append(path)
        return image_paths

    def _load_annotations(self) -> Tuple[list, list]:
        classes, coords = [], []
        images_root_path = os.path.join(self._root_path, self._image_set)

        label_type = self._get_label_type()
        classes_list = self.class_names

        for index, single_id in enumerate(self.image_ids):
            seq = single_id.split("/")[0]
            file = single_id.split("/")[1]
            file = file.replace("camera", label_type)
            file += ".json"

            ann_path = os.path.join(images_root_path, seq, label_type,
                                    "cam_front_center", file)

            image_width = self.img_widths[index]
            image_height = self.img_heights[index]
            with open(ann_path) as json_file:
                data = json.load(json_file)
                cl = [classes_list.index(data[box]["class"]) for box in
                      data.keys()]
                if len(cl) <= 0:
                    co = np.array([])
                else:
                    coord_lists = [data[box]["2d_bbox"] for box in data.keys()]
                    co = np.array(coord_lists)
                    co[co < 0] = 0
                    co[co[:, 0] > image_width, 0] = image_width
                    co[co[:, 1] > image_height, 1] = image_height
                    co[co[:, 2] > image_width, 2] = image_width
                    co[co[:, 3] > image_height, 3] = image_height
                    co = co.astype(int)

            classes.append(cl)
            coords.append(co)

        return classes, coords

    def _get_label_type(self):
        return "label3D"

    def _get_sequence_splits(self):
        return {
            "train": ["20181204_170238", "20181204_154421", "20181108_123750", "20181108_103155", "20181108_091945",
                      "20181107_133258", "20181107_132730", "20181107_132300", "20181016_125231", "20181008_095521",
                      "20180925_135056", "20180925_124435", "20180925_112730", "20180925_101535", "20180810_142822",
                      "20180807_145028"],
            "val": ["20181108_084007"],
            "test": ["20181204_135952"],
            "mini": ["mini"]
        }

    def _remove_bad_images(self):
        for index in reversed(range(len(self.image_ids))):
            co = self.coordinates[index]
            if co.size == 0:
                self._delete_sample_at(index)
            else:
                box_widths = co[:, 2] - co[:, 0]
                box_heights = co[:, 3] - co[:, 1]
                if 0.0 in box_widths or 0.0 in box_heights:
                    self._delete_sample_at(index)

    def _delete_sample_at(self, index):
        del self.classes[index]
        del self.coordinates[index]
        del self._image_paths[index]
        del self.img_widths[index]
        del self.img_heights[index]
        del self.image_ids[index]
