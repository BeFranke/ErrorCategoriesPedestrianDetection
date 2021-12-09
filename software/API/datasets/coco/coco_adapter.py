import json
import os
import tempfile
from typing import Tuple

from pycocotools.coco import COCO

from datasets.coco.coco_dataset import COCODataset
from datasets.base_dataset import BaseDataset


class COCOAdapter(COCODataset):
    """
    This adapter converts any dataset to a COCO compatible dataset. To achieve
    this task we need to convert the groundtruth annotations and image
    information to a COCO instances Json file. After that we can use the
    official COCO API to create an COCO object.
    """

    def __init__(self, dataset: 'BaseDataset') -> None:
        self._dataset = dataset
        super().__init__(dataset._config, dataset._image_set, dataset._mode, dataset.augmentation)

    def _load_coco_json_annotations(self, config, image_set, mode):
        annotations = []
        images = []
        categories = set()
        annotation_id = 1

        for img_index, image_id in enumerate(self._dataset.image_ids):
            for ann_index, annotation_class in enumerate(
                    self._dataset.classes[img_index]):
                coords = self._dataset.coordinates[img_index][ann_index]
                width = int(abs(coords[2] - coords[0]))
                height = int(abs(coords[3] - coords[1]))

                new_ann = {}
                new_ann["id"] = annotation_id
                new_ann["category_id"] = int(annotation_class)
                new_ann["iscrowd"] = 0
                new_ann["image_id"] = image_id
                new_ann["bbox"] = [int(coords[0]), int(coords[1]), width, height]
                new_ann["area"] = height * width
                new_ann["segmentation"] = []
                annotations.append(new_ann)
                annotation_id += 1

                categories.add(int(annotation_class))

            new_img = {}
            new_img["id"] = image_id
            new_img["width"] = int(self._dataset.img_widths[img_index])
            new_img["height"] = int(self._dataset.img_heights[img_index])

            images.append(new_img)

        final_json = {}
        final_json["images"] = images
        final_json["annotations"] = annotations
        final_json["categories"] = [{"id": cat_id} for cat_id in categories]

        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as tmp:
            json.dump(final_json, tmp)

        return COCO(path)

    def _get_dataset_root_path(self) -> str:
        return self._dataset._root_path

    def _load_image_ids(self) -> list:
        return self._dataset.image_ids

    def _load_image_sizes(self) -> Tuple[list, list]:
        return self._dataset.img_widths, self._dataset.img_heights

    def _load_image_paths(self) -> list:
        return self._dataset._image_paths

    def _load_annotations(self) -> Tuple[list, list]:
        return self._dataset.classes, self._dataset.coordinates

    def get_label_map(self):
        return {i: i for i in range(100)}

    def _get_image_set(self):
        return self._dataset._image_set

