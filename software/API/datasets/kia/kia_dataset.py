import json
import os
import yaml
import cv2
import torch

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as FT

from PIL import Image
from typing import Tuple

from patrick.datasets.base_dataset import BaseDataset

# CityPersons: 1024 x 2048
# KIA: 1280 x 1920

class KIADataset(BaseDataset):

    def __init__(self, config, image_set, mode, augmentation) -> None:
        super().__init__(config, image_set,  mode, augmentation)

    def _get_image_set(self):
        return self.image_set

    def _get_dataset_root_path(self) -> str:
        return self.config.kia_root

    def _get_class_names(self) -> list:
        return ['background', 'person']

    def _load_image_ids(self) -> list:
        sequences, ids = [], []
        images_root_path = self.root_path

        splits = self._get_sequence_splits()

        for seq in splits[self.image_set]:
            sequence_path = os.path.join(images_root_path, seq, "sensor/camera/left/png/")
            files = sorted(os.listdir(sequence_path))
            ids += [seq + "/" + name[:-4] for name in files if name.endswith('.png')]
        return ids

    def _load_image_sizes(self) -> Tuple[list, list]:
        img_widths, img_heights = [], []
        for path in self.image_paths:
            image = Image.open(path, mode='r')
            img_widths.append(image.width)
            img_heights.append(image.height)
        return img_widths, img_heights

    def _get_sequence_splits(self):
        kia_splits_path = './datasets/kia/kia_splits.yaml'
        with open(kia_splits_path, "r") as file:
            _splits = yaml.safe_load(file)

        # Use self.image_set:
        #   1) bit_tranche_2
        #   2) popc_tranche_3
        # Output has to be dict with (train, val, test, inference)
        splits = {'mini': _splits['mini'],
                  'train_tp1_tranche_4': _splits['train_tp1_tranche_4'],
                  'val_tp1_tranche_4': _splits['val_tp1_tranche_4'],
                  }

        # for split in ['train', 'val', 'test_a', 'test_b', 'test_c']:
        #     splits[split] = _splits[split + '_' + self.image_set]

        return splits

    def _load_image_paths(self) -> list:
        image_paths = []
        images_root_path = self.root_path
        for single_id in self.image_ids:
            seq = single_id.split("/")[0]
            file = single_id.split("/")[1] + ".png"
            path = os.path.join(images_root_path, seq,
                                "sensor/camera/left/png/", file)
            image_paths.append(path)
        return image_paths

    def _load_annotations(self) -> Tuple[list, list]:
        '''
        KIA comes with in [cx, cy, w, h] Format!

        return: coords in [xmin, ymin, xmax, ymax] Format
        '''

        cityscapesW, cityscapesH = 2048, 1024
        kiaW, kiaH = 1920, 1280
        rx = cityscapesW / kiaW
        ry = cityscapesH / kiaH

        classes, coords, instance_ids, vis_ratio, bbox, vis_bbox = [], [], [], [], [], []
        images_root_path = self.root_path
        classes_list = self.class_names
        self.del_samples = []

        for index, single_id in enumerate(self.image_ids):
            seq = single_id.split("/")[0]
            file = single_id.split("/")[1]
            file += ".json"

            ann_path = os.path.join(images_root_path, seq, "ground-truth/2d-bounding-box-fixed_json", file)

            with open(ann_path) as json_file:
                data = json.load(json_file)

                cl, co, _occl, instIds, _bbox, _vis_bbox, _vis_ratio = [], [], [], [], [], [], []
                for instId in data.keys():

                    if data[instId]["class_id"] == 'human' \
                            or data[instId]["class_id"] == 'person':
                        cl.append(classes_list.index('person'))

                        x1 = data[instId]["c_x"] - data[instId]["w"] / 2.0
                        y1 = data[instId]["c_y"] - data[instId]["h"] / 2.0
                        _box = np.array([int(rx * x1),
                                         int(ry * y1),
                                         int(rx * (data[instId]["c_x"] + data[instId]["w"] / 2.0)),
                                         int(ry * (data[instId]["c_y"] + data[instId]["h"] / 2.0))])
                        co.append(_box)
                        instIds.append(int(instId))

                        # _bbox in [xmin, ymin, w, h]
                        # _bbox.append([x1, y1, data[instId]["w"], data[instId]["h"]])
                        _bbox.append([rx * data[instId]['bbox_cityscapes'][0],
                                      ry * data[instId]['bbox_cityscapes'][1],
                                      rx * data[instId]['bbox_cityscapes'][2],
                                      ry * data[instId]['bbox_cityscapes'][3],
                                      ])
                        _vis_bbox.append([rx * data[instId]['vis_bbox'][0],
                                          ry * data[instId]['vis_bbox'][1],
                                          rx * data[instId]['vis_bbox'][2],
                                          ry * data[instId]['vis_bbox'][3],
                                          ])
                        _vis_bbox.append(data[instId]['vis_bbox'])
                        _vis_ratio.append(data[instId]['vis_ratio_bb'])

                    else:
                        continue

                co = np.array(co)
                cl = np.array(cl)

            classes.append(cl)
            coords.append(co)
            instance_ids.append(instIds)

            vis_ratio.append(_vis_ratio)
            bbox.append(_bbox)
            vis_bbox.append(_vis_bbox)

        self.instance_ids = instance_ids
        self.bbox = bbox
        self.vis_bbox = vis_bbox
        self.vis_ratio = vis_ratio

        return classes, coords

    def _delete_sample_at(self, index):
        del self.classes[index]
        del self.coordinates[index]
        del self.image_paths[index]
        del self.img_widths[index]
        del self.img_heights[index]
        del self.image_ids[index]
        del self.vis_ratio[index]
        del self.vis_bbox[index]
        del self.bbox[index]
        del self.instance_ids[index]

    def rescale_coords(self, array):

        cityscapesW, cityscapesH = 2048, 1024
        kiaW, kiaH = 1920, 1280
        rx = cityscapesW / kiaW
        ry = cityscapesH / kiaH

        array = array.astype(dtype='float32')

        array[:, [0, 2]] *= rx
        array[:, [1, 3]] *= ry

        return array.astype(dtype='int64')

    def load_segmentation(self, index, segmentationType='semantic-instance-segmentation_png'):

        file_path = self.image_ids[index]
        seq = file_path.split("/")[0]
        file = file_path.split("/")[1]

        inst_seg_path = os.path.join(self.root_path, seq,
                                'ground-truth',
                                segmentationType,
                                '{}.png'.format(file))

        segmentation = Image.open(inst_seg_path, mode='r')
        # plt.imshow(segmentation), plt.show()
        # _segmentation = np.transpose(np.array(segmentation), (2, 0, 1))
        segmentation = segmentation.resize(size=(self.config.img_res[self.mode][1], self.config.img_res[self.mode][0]),
                                           resample=Image.NEAREST)
        # __segmentation = np.transpose(np.array(segmentation), (2, 0, 1))
        # plt.imshow(segmentation), plt.show()
        return segmentation

    def get_instances(self, instance, instance_ids):
        '''
        Read instance as a list of torch Tensors.

        :param instance: PIL Image
        '''

        W = self.config.img_res[self.mode][1] // self.config.down_factor
        H = self.config.img_res[self.mode][0] // self.config.down_factor
        instance = instance.resize(size=(W, H), resample=Image.NEAREST)

        instance = np.array(instance, dtype='int')
        instance = instance[:,:,::-1]   # Instance Segmentation is in 'BGR'!
        instance = instance[:, :, 2] * 2 ** 16 + instance[:, :, 1] * 2 ** 8 + instance[:, :, 0]
        instance = torch.from_numpy(instance)

        instanceMasks = []
        for instId in instance_ids:
            instanceMask = instance == int(instId)
            instanceMasks.append(instanceMask.bool())

        if len(instanceMasks) > 0:
            instanceMasks = torch.stack(instanceMasks, dim=0)  # [num_gts, H, W]
        else:
            instanceMasks = None

        return instanceMasks

    def get_body_parts(self, bodyPart, instances):
        if bodyPart is None: return None

        # Decode Body Parts: https://confluence.vdali.de/pages/viewpage.action?pageId=4195324&preview=/4195324/4198994/KI-A_TP1_ERG_1.2.3%20Spezifikation%20des%20Annotationsformats_nightly.docx
        head = [
                # (158, 252, 35),  # Gesicht
                (115,205,115), # Gesicht ?!
                (170, 170, 170),  # Verdeckung Kopf
                (219, 219, 219),  # Haar
                (0, 185, 186),  # Linkes Auge
                (186, 0, 4),  # Rechtes Auge
                (118, 0, 186),  # Mund
                (186, 0, 182),  # Nase
        ]
        torso = [(70, 35, 252),  # Vorne Torso
                 (148, 35, 252),  # Hinten Torso
        ]
        arm = [(80,252,35),     # Linker Oberarm
               (252,70,35),     # Rechter Oberarm
               (35,252,177),    # Linker Unterarm
               (252,35,131),    # Rechter Unterarm
        ]
        hand = [(35,123,252),   # Linke Hand
                (225,35,252),   # Rechte Hand
        ]
        leg = [(162,225,116),  # Linker Oberschenkel
               (225,209,116),  # Rechter Oberschenkel
               (116,225,190),  # Linker Unterschenkel
               (225,127,116),  # Rechter Unterschenkel
               (178, 218, 105), # Verdeckung Beine
        ]
        foot = [(116,149,225), # Linker Fuß
                (225,116,222), # Rechter Fuß
        ]

        partDict = dict(head=head, torso=torso, arm=arm, hand=hand, leg=leg, foot=foot)

        W = self.config.img_res[self.mode][1] // self.config.down_factor
        H = self.config.img_res[self.mode][0] // self.config.down_factor
        bodyPart = bodyPart.resize(size=(W, H), resample=Image.NEAREST)
        # plt.imshow(bodyPart, interpolation='none'), plt.show()

        bodyPart = np.array(bodyPart, dtype='int')
        # _bodyPart = np.transpose(bodyPart, (2, 0, 1))
        bodyPart = torch.from_numpy(bodyPart)   # [H, W, 3]

        bodyParts = []
        if instances is not None:
            for instance_id in range(instances.size(0)):
                instance = instances[instance_id, :, :]
                mask = instance > 0

                bodyPartInstance = bodyPart * mask.long()[:, :, None]   # [H, W, 3]
                # _bodyPartInstance = bodyPartInstance.permute(2, 0, 1).numpy()
                # plt.imshow(bodyPartInstance.numpy(), interpolation='none'), plt.show()

                separatedBodyParts = []
                for key, value in partDict.items():
                    mask = torch.zeros((H*W), dtype=torch.uint8)
                    for rgb in value:
                        rgbTensor = torch.ones((H, W, 3), dtype=torch.int64) * torch.tensor(rgb, dtype=torch.int64)
                        partMask = bodyPartInstance == rgbTensor
                        valid = partMask.sum(dim=2).view(-1) == 3
                        mask += valid

                    # if mask.sum() > 1:
                    #     plt.imshow(mask.view(H, W).long().numpy(), interpolation='none', cmap='Greys'), plt.show()
                    #     print(key)

                    separatedBodyParts.append(mask.view(H, W).bool())
                bodyParts.append(torch.stack(separatedBodyParts))
            bodyParts = torch.stack(bodyParts)  # [num_instances, num_body_parts, H, W]

        else:
            bodyParts = None

        return bodyParts

    def get_target(self, image, classes, coords, image_path, index):
        '''
        :param coords: minmax Format for Cityscapes resolution
        '''

        # Load Image: KIA Size has to be changed for CityPerson validation resolution
        image = image.resize(size=(self.config.img_res['val'][1],
                                   self.config.img_res['val'][0]),
                             resample=Image.BILINEAR)
        # plt.imshow(image), plt.show()

        # Load Instance Semantic Segmentation
        needBodyPart = False
        instance = self.load_segmentation(index=index, segmentationType='semantic-instance-segmentation_png')
        bodyPart = self.load_segmentation(index=index, segmentationType='body-part-segmentation_png') if needBodyPart else None
        instance_ids = torch.tensor(self.instance_ids[index])

        if self.mode == 'train':
            if self._augmentation:
                image, coords, classes, instance, instance_ids, bodyPart = self.augmentation(image, classes, coords,
                                                                                             instance,
                                                                                             instance_ids,
                                                                                             bodyPart)

            else:
                ratio = self.config.img_res['train'][0] / self.config.img_res['val'][0]
                coords = coords.clone() * ratio

                image = image.resize(size=(self.config.img_res['train'][1],
                                           self.config.img_res['train'][0]),
                                     resample=Image.BILINEAR)

        instances = self.get_instances(instance=instance, instance_ids=instance_ids)
        bodyParts = self.get_body_parts(bodyPart=bodyPart, instances=instances)

        # plt.imshow(image), plt.show()
        # plt.imshow(instance), plt.show()
        # plt.imshow(bodyPart), plt.show()

        target = dict(image=image, classes=classes, coords=coords, image_path=image_path,
                      instances=instances, bodyParts=bodyParts)

        return image, target


def visualize():
    import importlib
    import os
    import torch

    from PIL import Image
    from torch.utils import data

    from trainer.trainer_builder import TrainerBuilder

    os.chdir('../../')
    print(os.getcwd())

    trainer = TrainerBuilder() \
        .update(config_root='./cfg_train.yaml') \
        .create()
    trainer.config.augmentation = False

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
    # i = 'bit_results_sequence_0147-4d53650ffc4a49909671fd74ed6beec0/car-camera004-0147-4d53650ffc4a49909671fd74ed6beec0-0250'
    i = 'bit_results_sequence_0179-21032ac691f24ce087ab3c4cc3a0b5fc/arb-camera052-0179-21032ac691f24ce087ab3c4cc3a0b5fc-0051'
    j, k = dataset.image_ids.index(i), 0

    for id in range(j, j+1):
        x, target = dataset.__getitem__(index=id)

        fig, ax = evaluator_inference.plot_image(image=target['image'])
        evaluator_inference.plot_gt_with_height(classes=target['classes'],
                                                coords=target['coords'],
                                                ax=ax)
        plt.show()
        print(dataset.image_ids[id])

        # for i in range(target['instances'].size(0)):
        #     plt.imshow(target['instances'][i, :, :].numpy())
        #     plt.show()


if __name__ == '__main__':

    visualize()
