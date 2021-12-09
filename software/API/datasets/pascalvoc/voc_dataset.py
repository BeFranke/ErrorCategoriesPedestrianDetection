import os
from typing import Tuple

import numpy as np
from bs4 import BeautifulSoup

from datasets.base_dataset import BaseDataset


class VOCDataset(BaseDataset):
    """
    This class represents the data for a PascalVOC dataset. The class inherits
    from the 'Dataset' class and is compatible to all the training, inference
    and evaluation scripts.
    """

    def __init__(self, config, image_set, mode, augmentation) -> None:
        super().__init__(config, image_set, mode, augmentation)
        self.difficulties = self._load_difficulties()

    def _get_image_set(self):
        return self.image_set

    def _get_dataset_root_path(self) -> str:
        return self.config.pascal_voc_root

    def _load_image_ids(self) -> list:
        ids = []
        image_set_path = os.path.join(self.root_path, self.image_set,
                                      'ImageSets/Main/{}.txt'.format(
                                          self._mode))
        with open(image_set_path) as file:
            ids += [line.strip() for line in file]

        return ids

    def _load_image_sizes(self) -> Tuple[list, list]:
        img_widths, img_heights = [], []
        ann_path = os.path.join(self.root_path, self.image_set,
                                "Annotations/")

        for single_id in self.image_ids:
            with open(ann_path + single_id + '.xml') as file:
                soup = BeautifulSoup(file, 'xml')
                sizes = soup.find_all('size')
                for size in sizes:
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)
            img_widths.append(width)
            img_heights.append(height)
        return img_widths, img_heights

    def _load_image_paths(self) -> list:
        images_root_path = os.path.join(self.root_path, self.image_set,
                                        "JPEGImages/")

        image_paths = []

        for single_id in self.image_ids:
            image_file_name = "{}.jpg".format(single_id)
            path = os.path.join(images_root_path, image_file_name)
            image_paths.append(path)
        return image_paths

    def _load_annotations(self) -> Tuple[list, list]:
        classes, coords = [], []
        ann_path = os.path.join(self.root_path, self.image_set,
                                "Annotations/")

        for single_id in self.image_ids:
            with open(ann_path + single_id + '.xml') as file:
                soup = BeautifulSoup(file, 'xml')
                cl, co = self._load_bounding_box_information(soup, single_id)
                classes.append(np.array(cl))
                coords.append(np.array(co))

        return classes, coords

    # Pascal VOC dataset specifc methods
    def _load_difficulties(self) -> list:
        """
        Retrieves the difficulties for each of the bounding boxes within this
        PascalVOC dataset.

        :return: A list of lists with booleans that indicate the difficulty for
            each of the bounding boxes. The first dimension belongs to the
            images while the second dimension belongs to the detections per
            image.
        """
        difficulties = []
        ann_path = os.path.join(self.root_path, self.image_set,
                                "Annotations/")

        for single_id in self.image_ids:
            with open(ann_path + single_id + '.xml') as file:
                dif = []
                soup = BeautifulSoup(file, 'xml')
                objects = soup.find_all('object')
                for obj in objects:
                    dif += [(obj.find('difficult').text).lower() in (
                        "yes", "true", "t", "1")]
                difficulties.append(dif)
        return difficulties

    def _load_bounding_box_information(self, soup, img_id):
        """
        Loads and returns all the bounding box information for the image with
        the given id.

        :param soup: The BeatifulSoup object that contains the content of the
            annotation xml file.
        :param img_id: The id of the image that belongs to the soup object.

        :return: The list of classes and coordinates for the image
        """
        classes, coords = [], []

        objects = soup.find_all('object')

        for obj in objects:
            class_name = obj.find('name', recursive=False).text
            class_id = self.class_names.index(class_name)
            bndbox = obj.find('bndbox', recursive=False)

            # -1 because of different base index for pascal voc
            item_dict = {'image_id': img_id,
                         'class_name': class_name,
                         'class_id': class_id,
                         'xmin': int(bndbox.xmin.text) - 1,
                         'ymin': int(bndbox.ymin.text) - 1,
                         'xmax': int(bndbox.xmax.text) - 1,
                         'ymax': int(bndbox.ymax.text) - 1}

            ordered_coordinates = [item_dict[item] for item in
                                   ['xmin', 'ymin', 'xmax', 'ymax']]
            coords.append(ordered_coordinates)
            classes.append(class_id)

        return classes, coords

    def _get_class_names(self) -> list:
        return ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor']
