from copy import deepcopy
from typing import Tuple, List, Dict

import numpy as np

from patrick.datasets.base_dataset import BaseDataset


class FusionDataset(BaseDataset):

    def __init__(self, config, datasets: List[BaseDataset], mode: str, augmentation: bool) -> None:
        self._datasets = datasets
        self._index_lookup, self.class_names = self._create_index_lookup()
        self._original_image_ids = self._load_original_image_ids()
        super().__init__(config=config, image_set="fusion_dataset", mode=mode, augmentation=augmentation)

    def _get_image_set(self):
        return "fusion_dataset"

    def _create_index_lookup(self) -> Tuple[Dict[int, List], List[str]]:
        category_id_lookup = {}
        class_name_lookup = {}
        final_class_names = []

        for index, dataset in enumerate(self._datasets):
            names = dataset.class_names
            category_ids = list(range(len(dataset.class_names)))
            category_id_lookup[index] = category_ids
            class_name_lookup[index] = [self._standardize(s) for s in names]

        name_counter = 0

        for dataset_index in class_name_lookup.keys():
            for name_index, name in enumerate(class_name_lookup[dataset_index]):
                if name in final_class_names:
                    new_index = final_class_names.index(name)
                    category_id_lookup[dataset_index][name_index] = new_index
                else:
                    final_class_names.append(name)
                    category_id_lookup[dataset_index][name_index] = name_counter
                    name_counter += 1

        return category_id_lookup, final_class_names

    def _standardize(self, name):
        name = name.lower()
        name = name.replace(" ", "")
        return name

    def _get_dataset_root_path(self) -> str:
        return ";".join([d.root_path for d in self._datasets])

    def _get_class_names(self) -> List[str]:
        return self.class_names

    def _load_image_ids(self) -> List:
        return list(range(len(self._original_image_ids)))

    def _load_original_image_ids(self) -> List:
        original_ids = []
        for dataset in self._datasets:
            original_ids += dataset.image_ids
        return original_ids

    def _load_image_sizes(self) -> Tuple[List[int], List[int]]:
        widths, heights = [], []
        for dataset in self._datasets:
            widths += dataset.img_widths
            heights += dataset.img_heights
        return widths, heights

    def _load_image_paths(self) -> List[str]:
        image_paths = []
        for dataset in self._datasets:
            image_paths += dataset.image_paths
        return image_paths

    def _load_annotations(self) -> Tuple[List, List]:
        classes = []
        coordinates = []

        for key in self._index_lookup:
            dataset = self._datasets[key]
            old_classes = deepcopy(dataset.classes)
            new_classes = deepcopy(dataset.classes)
            for old_id, new_id in enumerate(self._index_lookup[key]):
                for class_index, class_list in enumerate(new_classes):
                    old_list = old_classes[class_index]
                    indices = np.argwhere(old_list == old_id)
                    if indices.size != 0:
                        class_list[indices] = new_id
            classes.extend(new_classes)
            coordinates.extend(dataset.coordinates)

        return classes, coordinates

