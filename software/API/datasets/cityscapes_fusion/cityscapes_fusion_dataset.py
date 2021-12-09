import json
import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

import torch
from torchvision import transforms as tr

from benedikt import debug
from patrick.datasets.base_dataset import BaseDataset


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CityscapesFusionDataset(BaseDataset):

    def __init__(self, config, image_set, mode, augmentation) -> None:
        super().__init__(config, image_set, mode, False)
        self.add_aug = augmentation

    def _get_image_set(self):
        return self.image_set

    def _get_dataset_root_path(self) -> str:
        return self.config.cityscapes_fusion_root

    def _get_class_names(self) -> list:
        # So far, ignore other classes
        return ['ignore', 'pedestrian', 'rider', 'sitting person', 'person (other)', 'person group']

    def _load_image_ids(self) -> list:
        self.splits = self._get_sequence_splits()

        inter_path = os.path.join(self.root_path, 'leftImg8bit/{}/'.format(self.splits[self.image_set]))

        subdirs = [x[0] for x in os.walk(inter_path)]
        subdirs.pop(0)
        subdirs = sorted(subdirs)

        ids = []
        for subdir in subdirs:
            id = sorted(os.listdir(subdir))
            city = subdir.split("/")[-1]
            id = [os.path.join(city, i) for i in id]
            ids += id

        # Remove '.png'
        ids_cut = [i[:-4] for i in ids]
        return ids_cut

    def _load_image_sizes(self) -> Tuple[list, list]:
        img_widths, img_heights = [], []
        for path in self.image_paths:
            image = torch.load(path)
            img_widths.append(image.shape[3])
            img_heights.append(image.shape[2])
        return img_widths, img_heights

    def _load_image_paths(self) -> list:
        image_paths = []
        images_root_path = os.path.join(self.root_path, 'leftImg8bit/{}/'.format(self.splits[self.image_set]))
        subdirs = [x[0] for x in os.walk(images_root_path)]
        subdirs.pop(0)
        subdirs = sorted(subdirs)

        for subdir in subdirs:
            id = sorted(os.listdir(subdir))
            path = [os.path.join(subdir, i) for i in id]
            image_paths += path
        return image_paths

    def _get_sequence_splits(self):
        return {
            'train': 'train',
            'test': 'test',
            'val': 'val',
            'mini': 'mini',
            'midi': 'midi',
            'micro': 'micro',
            'macro': 'macro',
            'duo': 'duo',
        }

    def _load_annotations(self) -> Tuple[list, list]:
        '''
        CityPersons comes with in [xmin, ymin, w, h] Format!

        return: coords in [xmin, ymin, xmax, ymax] Format
        '''

        classes, coords, vis_ratio, bbox, vis_bbox = [], [], [], [], []
        annotation_root_path = os.path.join(self.root_path,
                                            'gtBboxCityPersons/{}/'
                                            .format(self.splits[self.image_set]))
        classes_list = self.class_names

        for index, single_id in enumerate(self.image_ids):
            single_id = single_id[:-12]
            ann_path = os.path.join(annotation_root_path, '{}_gtBboxCityPersons.json'.format(single_id))

            with open(ann_path) as json_file:
                data = json.load(json_file)

                _classes, _coords, _occlusion, _bbox, _vis_bbox, _vis_ratio = [], [], [], [], [], []
                for obj in data['objects']:

                    _bbox.append(obj['bbox'])
                    _vis_bbox.append(obj['bboxVis'])

                    x1, y1, w, h = obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]
                    wBboxVis, hBboxVis = obj['bboxVis'][2], obj['bboxVis'][3]

                    x1, y1 = max(int(x1), 0), max(int(y1), 0)
                    w, h = min(int(w), self.img_widths[0] - x1 - 1), min(int(h), self.img_heights[0] - y1 - 1)
                    box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])

                    _classes += [classes_list.index(obj['label'])]
                    _coords.append(box)

                    _vis_ratio.append((wBboxVis * hBboxVis) / (w * h))

            coords.append(np.array(_coords).astype(int))
            classes.append(np.array(_classes))

            vis_ratio.append(_vis_ratio)
            bbox.append(_bbox)
            vis_bbox.append(_vis_bbox)

        self.bbox = bbox
        self.vis_bbox = vis_bbox
        self.vis_ratio = vis_ratio
        return classes, coords

    def get_target(self, image, classes, coords, image_path, index):
        new_height, new_width = self.config.img_res['train'][0], self.config.img_res['train'][1]
        # image = image.resize(size=(new_width, new_height), resample=Image.BILINEAR)
        transforms = [tr.Resize((new_width, new_height))]
        if self.add_aug:
            transforms += [AddGaussianNoise]
        image = tr.Compose(transforms)(image)
        ratio = self.config.img_res['train'][0] / self.config.img_res['val'][0]
        coords = coords.clone() * ratio

        target = dict(image=image, classes=classes, coords=coords, image_path=image_path)
        return image, target

    def __getitem__(self, index):
        """
        Loads an image (input) and its class and coordinates (targets) based on
        the index within the list of image ids. That means that the image
        information that will be returned belongs to the image at given index.

        :param index: The index of the image within the list of image ids that
                you want to get the information for.

        :return: A quadruple of an image, its class, its coordinates and the
                path to the image itself.
        """
        image_path = self.image_paths[index]
        debug.OD_PATH = image_path
        image = torch.load(image_path)

        classes = deepcopy(self.classes[index])
        coords = deepcopy(self.coordinates[index])
        classes = torch.tensor(classes, dtype=torch.long)
        coords = torch.tensor(coords, dtype=torch.float32)

        image, target = self.get_target(image=image, classes=classes, coords=coords, image_path=image_path, index=index)

        # Tensor of Input Image
        # image_tensor = FT.to_tensor(image)
        # image_tensor = FT.normalize(image_tensor, mean=self.config.norm_mean, std=self.config.norm_std)

        return image, target
