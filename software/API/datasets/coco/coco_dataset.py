import os
from typing import Tuple

import numpy as np
from pycocotools.coco import COCO

from datasets.base_dataset import BaseDataset


class COCODataset(BaseDataset):
    """
    This class represents the data for a COCO dataset. The class inherits
    from the 'Dataset' class and is compatible to all the training, inference
    and evaluation scripts.

    To get the bounding box annotations we use the offical COCO API to read the
    instances.json file.

    Furthermore, we need to remove some images from the dataset because they
    might not have a single detection or they might have detections with a zero
    area. Pleas take a look at the filter_images.py file to generate an images
    blacklist.
    """

    def __init__(self, config, image_set, mode, augmentation) -> None:
        self.coco_annotations = self._load_coco_json_annotations(config, image_set, mode)
        super().__init__(config, image_set, mode, augmentation)

    def _get_image_set(self):
        return self._image_set

    def _load_coco_json_annotations(self, config, image_set, mode):
        ann_path = 'annotations/instances_{}{}.json'.format(mode, image_set)
        return COCO(os.path.join(config.coco_root, image_set, ann_path))

    def _get_dataset_root_path(self) -> str:
        return self._config.coco_root

    def _load_image_ids(self) -> list:
        image_ids = list(sorted(self.coco_annotations.imgs.keys()))

        blacklist_template = 'blacklists/img_ids_blacklist_{}{}.txt'
        blacklist_file = blacklist_template.format(self._mode, self._image_set)
        blacklist_path = os.path.join(self._root_path, self._image_set,
                                      blacklist_file)

        image_ids = self._remove_blacklisted_images(blacklist_path, image_ids)

        return image_ids

    def _load_image_sizes(self) -> Tuple[list, list]:
        img_widths, img_heights = [], []
        for single_id in self.image_ids:
            image_information = self.coco_annotations.loadImgs(single_id)[0]
            img_widths.append(image_information["width"])
            img_heights.append(image_information["height"])
        return img_widths, img_heights

    def _load_image_paths(self) -> list:
        image_folder_name = "{}{}/".format(self._mode, self._image_set)
        images_root_path = os.path.join(self._root_path, self._image_set,
                                        image_folder_name)

        image_paths = []

        for single_id in self.image_ids:
            image_name = self.coco_annotations.loadImgs(single_id)[0][
                'file_name']
            path = os.path.join(images_root_path, image_name)
            image_paths.append(path)
        return image_paths

    def _get_class_names(self) -> list:
        return ['background', 'person', 'bicycle', 'car', 'motorcycle',
                'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

    def _load_annotations(self) -> Tuple[list, list]:
        classes = []
        coords = []
        label_map = self.get_label_map()

        for single_id in self.image_ids:
            annotations_ids = self.coco_annotations.getAnnIds(imgIds=single_id)
            targets = self.coco_annotations.loadAnns(annotations_ids)

            # Read the class and the coordinates within the list of labels
            labels = self._reindex(targets, label_map)
            classes.append(labels[:, 0])
            coords.append(labels[:, 1:])
        return classes, coords

    # COCO dataset specifc methods
    def _remove_blacklisted_images(self, blacklist_path, image_ids):
        """
        Removes all the blacklisted train2017 from the list of given image_ids.
        A blacklisted image is an image that might not have proper annotations
        or is not suitable for training for any kind of reason. To remove all
        blacklisted train2017 the user of this function needs to specify a path
        to a file that contains the blacklisted image ids.

        :param blacklist_path: The path to the file that contains the
            blacklisted image ids for the dataset at hand.
        :param image_ids: The list of all image ids that should be filtered.

        :return: A list of image ids that does not contain any of the
            blacklisted image ids.
        """
        if os.path.isfile(blacklist_path):
            with open(blacklist_path, 'r') as blacklist:
                for line in blacklist:
                    line = line.replace(" ", "")
                    line = line.strip()
                    line = line.strip(",")
                    blacklisted_ids = line.split(',')
                    for bad_id in blacklisted_ids:
                        bad_id = int(bad_id)
                        if bad_id in image_ids:
                            image_ids.remove(bad_id)
        return image_ids

    def _reindex(self, target, label_map):
        """
        Changes the category ids within the COCO dataset to contiguos values.
        Right now some categories never appear in the whole COCO dataset. This
        means that an image with category 20 (sheep) becomes 19 for training.
        You can lookup the mapping in the coco_labels.txt file within the coco
        root folder.

        :param target: The target object from the COCO api.
        :param label_map: The map that contains the mapping from category_id to
            category_id.

        :return: An array of labels: The first column contains the classes while
            the seocnd column contains the coordinates of the bounding boxes.
        """
        res = []
        for obj in target:
            # This bbox as the format (xmin, ymin, width, height)
            bbox = obj['bbox']
            # This bbox as the format (xmin, ymin, xmax, ymax)
            bbox_minmax = [bbox[0], bbox[1], bbox[0] + bbox[2],
                           bbox[1] + bbox[3]]
            # Lookup the correct label for training
            label_idx = label_map[obj['category_id']]
            res.append(
                [label_idx, bbox_minmax[0], bbox_minmax[1], bbox_minmax[2],
                 bbox_minmax[3]])

        labels = np.array(res)

        if len(labels.shape) == 1:
            print('no box problem...')

        return labels

    def get_label_map(self):
        label_map = {}

        labels_file = 'coco_labels.txt'
        label_map_file = os.path.join(self._root_path, labels_file)

        labels = open(label_map_file, 'r')
        for line in labels:
            ids = line.split(',')
            label_map[int(ids[0])] = int(ids[1])

        return label_map
