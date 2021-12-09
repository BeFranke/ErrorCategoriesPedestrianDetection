import json
import os

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from PIL import Image

from patrick.datasets.base_dataset import BaseDataset


class CityscapesDataset(BaseDataset):

    def __init__(self, config, image_set, mode, augmentation, training=False) -> None:
        self.training = training
        super().__init__(config, image_set, mode, augmentation)
        self.seg_base_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "input", "datasets",
                                          "cityscapes", "gtFine", image_set)

    def _get_image_set(self):
        return self.image_set

    def _get_dataset_root_path(self) -> str:
        return self.config.cityscapes_root

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
            image = Image.open(path, mode='r')
            img_widths.append(image.width)
            img_heights.append(image.height)
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

        classes, coords, vis_ratio, bbox, vis_bbox, instance_ids = [], [], [], [], [], []
        annotation_root_path = os.path.join(self.root_path,
                                            'gtBboxCityPersons/{}/'
                                            .format(self.splits[self.image_set]))
        classes_list = self.class_names

        for index, single_id in enumerate(self.image_ids):
            single_id = single_id[:-12]
            ann_path = os.path.join(annotation_root_path, '{}_gtBboxCityPersons.json'.format(single_id))

            with open(ann_path) as json_file:
                data = json.load(json_file)

                _classes, _coords, _occlusion, _bbox, _vis_bbox, _vis_ratio, _instance_ids = [], [], [], [], [], [], []
                for obj in data['objects']:

                    _bbox.append(obj['bbox'])
                    _vis_bbox.append(obj['bboxVis'])

                    x1, y1, w, h = obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]
                    wBboxVis, hBboxVis = obj['bboxVis'][2], obj['bboxVis'][3]
                    assert wBboxVis <= w and hBboxVis <= h

                    x1, y1 = max(int(x1), 0), max(int(y1), 0)
                    if self.training:
                        wBboxVis, hBboxVis = min(int(wBboxVis), self.img_widths[0] - x1 - 1), min(int(hBboxVis), self.img_heights[0] - y1 - 1)
                        w, h = min(int(w), self.img_widths[0] - x1 - 1), min(int(h), self.img_heights[0] - y1 - 1)

                    box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])

                    _classes += [classes_list.index(obj['label'])]
                    _coords.append(box)

                    _vis_ratio.append((wBboxVis * hBboxVis) / (w * h))
                    # if round(_vis_ratio[-1], ndigits=12) == 0.811258278146:
                    #     assert False
                    if not self.training:
                        assert (wBboxVis * hBboxVis) / (w * h) <= 1
                    _instance_ids.append(int(obj['instanceId']))

            coords.append(np.array(_coords).astype(int))
            classes.append(np.array(_classes))
            instance_ids.append(_instance_ids)

            vis_ratio.append(_vis_ratio)
            bbox.append(_bbox)
            vis_bbox.append(_vis_bbox)

        self.bbox = bbox
        self.vis_bbox = vis_bbox
        self.vis_ratio = vis_ratio
        self.instance_ids = np.asarray(instance_ids)
        return classes, coords

    def get_target(self, image, classes, coords, image_path, index):

        if self.mode == 'train':
            if self._augmentation:
                image, coords, classes, _, _, _ = self.augmentation(image, classes, coords)

            else:
                new_height, new_width = self.config.img_res['train'][0], self.config.img_res['train'][1]
                image = image.resize(size=(new_width, new_height), resample=Image.BILINEAR)
                ratio = self.config.img_res['train'][0] / self.config.img_res['val'][0]
                coords = coords.clone() * ratio

        target = dict(image=image, classes=classes, coords=coords, image_path=image_path)
        return image, target


def visualize_gt():
    import importlib
    import os
    import torch
    os.chdir('../../')

    from PIL import Image
    from torch.utils import data

    from trainer.trainer_builder import TrainerBuilder
    from inferencer.utils import plot_output

    trainer = TrainerBuilder() \
        .update(config_root='./cfg_train.yaml') \
        .create()
    trainer.config.augmentation = True

    # Load/ import models dynamically
    module = importlib.import_module('models.{}.{}.{}'.format(trainer.config.model[:-2],
                                                              trainer.config.model,
                                                              trainer.config.model))
    model = getattr(module, trainer.config.model.upper())
    model = model(trainer.config, trainer.device)
    net_config = model.get_net_config()
    trainer.update_config(new_config=net_config)

    # Load Datasets and Tools
    trainer.config.batch_size = 1
    trainer.prepare()

    dataset = trainer.dataset_train

    # Load EvaluatorInference
    from evaluator.evaluator_inference import EvaluatorInference

    evaluator_inference = EvaluatorInference(config=trainer.config,
                                             device=trainer.device,
                                             dataset=dataset,
                                             mode=None)

    for id in range(0, 5):
        x, target = dataset.__getitem__(index=id)

        fig, ax = evaluator_inference.plot_image(image=target['image'])
        evaluator_inference.plot_gt_with_height(classes=target['classes'],
                                                coords=target['coords'],
                                                ax=ax)
        plt.show()


if __name__ == '__main__':

    visualize_gt()

