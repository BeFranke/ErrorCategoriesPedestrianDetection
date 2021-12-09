from typing import Tuple

from datasets.base_dataset import BaseDataset
from datasets.pascalvoc.voc_dataset import VOCDataset


class VOCAdapter(VOCDataset):
    """
    This adapter converts any dataset to a PascalVOC like dataset. This is
    particularly helpful if you want to evaluate a dataset like COCO or A2D2
    with the PascalVOC metric.
    """

    def __init__(self, dataset: 'BaseDataset') -> None:
        self._dataset = dataset
        super().__init__(dataset.config, dataset.image_set, dataset.mode, dataset.augmentation)

    def _load_difficulties(self):
        difficulties = []
        try:
            difficulties = self._dataset.difficulties
        except AttributeError:
            for index in range(len(self.image_ids)):
                # 1) None GT is difficult to detect
                # difficulties.append([False]*len(self.classes[index]))
                # 2) All ignore areas (class = 0) are difficult to detect!
                difficulties.append((1 - self.classes[index]).astype(bool).tolist())

        return difficulties

    def _get_dataset_root_path(self) -> str:
        return self._dataset.root_path

    def _load_image_ids(self) -> list:
        return self._dataset.image_ids

    def _load_image_sizes(self) -> Tuple[list, list]:
        return self._dataset.img_widths, self._dataset.img_heights

    def _load_image_paths(self) -> list:
        return self._dataset.image_paths

    def _load_annotations(self) -> Tuple[list, list]:
        return self._dataset.classes, self._dataset.coordinates

    def _get_image_set(self):
        return self._dataset.image_set