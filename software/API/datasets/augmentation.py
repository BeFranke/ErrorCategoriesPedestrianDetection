import torch
import torchvision

import torchvision.transforms.functional as FT
from torchvision.transforms import ColorJitter
import numpy as np

from copy import deepcopy
from PIL import Image

from patrick.models.lib.utils import calc_iou_tensor


class CSPAugmentation():

    def __init__(self, config):
        self.config = config

    @staticmethod
    def random_color_distortions(image):
        '''

        :param image: PIL Image
        :return:
        '''
        factor = (0.5, 2.0)

        photometric_distort = [
            (FT.adjust_brightness, factor),
            (FT.adjust_contrast, factor),
            (FT.adjust_saturation, factor),
            (FT.adjust_hue, (-18 / 255, 18 / 255))
        ]

        np.random.shuffle(photometric_distort)
        for distort, (_low, _high) in photometric_distort:
            if np.random.randint(0, 2) == 0:
                rand_factor = np.random.uniform(low=_low, high=_high)
                image = distort(image, rand_factor)

        return image

    @staticmethod
    def horizontal_flip(image, coords, instance=None, bodyPart=None):
        '''

        :param image: PIL Image
        :param coords: torch_tensor as [xmin, ymin, xmax, ymax]
        :param instance: None or PIL Image
        :return:
        '''
        image = FT.hflip(image)

        if instance is not None:
            instance = FT.hflip(instance)
        if bodyPart is not None:
            bodyPart = FT.hflip(bodyPart)

        if not coords.size(0) == 0:
            coords[:, [0, 2]] = image.width - coords[:, [2, 0]]

        return image, coords, instance, bodyPart

    def augment_csp(self, image, coords, classes, instance, instance_ids, bodyPart):
        '''

        :param image: PIL Image
        :param instance: None or PIL Image
        :param coords:
        :param classes:
        :param scale:
        :return:
        '''

        # 1) Resizing with Scale
        height, width = image.height, image.width
        ratio = np.random.uniform(0.4, 1.5)
        new_height, new_width = int(ratio * height), int(ratio * width)

        # Resizing of PIL Image is done by using size=(W, H)
        image = image.resize(size=(new_width, new_height), resample=Image.BILINEAR)
        if instance is not None:
            instance = instance.resize(size=(new_width, new_height), resample=Image.NEAREST)
        if bodyPart is not None:
            bodyPart = bodyPart.resize(size=(new_width, new_height), resample=Image.NEAREST)

        coords = coords.clone()
        classes = classes.clone()

        # Resize coords
        coords *= ratio

        # 2) Random Crop
        if image.height >= self.config.img_res['train'][0]:
            image, coords, classes, instance, instance_ids, bodyPart = self.random_crop_csp(image, coords, classes,
                                                                                            instance,
                                                                                            instance_ids, bodyPart,
                                                                                            crop_size=
                                                                                            self.config.img_res[
                                                                                                'train'],
                                                                                            limit=16)

        # 3) Random Pave
        else:
            image, coords, classes, instance, instance_ids, bodyPart = self.random_pave_csp(image, coords, classes,
                                                                                            instance,
                                                                                            instance_ids, bodyPart,
                                                                                            pave_size=
                                                                                            self.config.img_res[
                                                                                                'train'],
                                                                                            limit=16)


        return image, coords, classes, instance, instance_ids, bodyPart

    def random_crop_csp(self, image, coords, classes, instance, instance_ids, bodyPart, crop_size, limit=8):
        img_height, img_width = image.height, image.width
        crop_h, crop_w = crop_size

        if classes.sum() > 0:
            _sel_id = np.random.randint(0, int(classes.sum()))

            _pos = classes == 1
            idx = torch.nonzero(_pos)
            sel_id = idx[_sel_id]

            sel_center_x = int((coords[sel_id, 0] + coords[sel_id, 2]) / 2.0)
            sel_center_y = int((coords[sel_id, 1] + coords[sel_id, 3]) / 2.0)
        else:
            sel_center_x = int(np.random.randint(0, img_width - crop_w + 1) + crop_w * 0.5)
            sel_center_y = int(np.random.randint(0, img_height - crop_h + 1) + crop_h * 0.5)

        crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
        crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
        diff_x = max(crop_x1 + crop_w - img_width, int(0))
        crop_x1 -= diff_x
        diff_y = max(crop_y1 + crop_h - img_height, int(0))
        crop_y1 -= diff_y

        # cropped_image = np.copy(image[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w])
        cropped_image = image.crop((crop_x1, crop_y1, crop_x1 + crop_w, crop_y1 + crop_h))
        if instance is not None:
            cropped_instance = instance.crop((crop_x1, crop_y1, crop_x1 + crop_w, crop_y1 + crop_h))
        else:
            cropped_instance = None

        if bodyPart is not None:
            cropped_bodyPart = bodyPart.crop((crop_x1, crop_y1, crop_x1 + crop_w, crop_y1 + crop_h))
        else:
            cropped_bodyPart = None

        # crop detections
        if coords.size(0) > 0:
            before_area = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
            coords[:, [0, 2]] -= crop_x1
            coords[:, [1, 3]] -= crop_y1
            coords[:, [0, 2]] = np.clip(coords[:,[0, 2]], 0, crop_w)
            coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, crop_h)
            after_area = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])

            keep_idx_igs = ((coords[:, 2] - coords[:, 0]) >= 8) \
                           * ((coords[:, 3] - coords[:, 1]) >= 8) \
                           * (classes == 0)
            keep_idx_gts = ((coords[:, 2] - coords[:, 0]) >= limit) \
                           * (after_area >= 0.5 * before_area) \
                           * (classes == 1)

            coords = coords[keep_idx_gts + keep_idx_igs, :]
            classes = classes[keep_idx_gts + keep_idx_igs]

            if instance_ids is not None:
                instance_ids = instance_ids[keep_idx_gts + keep_idx_igs]

        return cropped_image, coords, classes, cropped_instance, instance_ids, cropped_bodyPart

    def random_pave_csp(self, image, coords, classes, instance, instance_ids, bodyPart, pave_size, limit=8):
        img_height, img_width = image.height, image.width
        pave_h, pave_w = pave_size

        # paved_image = np.zeros((pave_h, pave_w, 3), dtype=image.dtype)
        paved_image = np.ones((pave_h, pave_w, 3), dtype=np.array(image).dtype) * np.mean(np.array(image), dtype=int)
        pave_x = int(np.random.randint(0, pave_w - img_width + 1))
        pave_y = int(np.random.randint(0, pave_h - img_height + 1))
        paved_image[pave_y:pave_y + img_height, pave_x:pave_x + img_width] = image
        paved_image = Image.fromarray(np.uint8(paved_image)).convert('RGB')

        if instance is not None:
            paved_instance = np.zeros((pave_h, pave_w, 3), dtype=int)
            paved_instance[pave_y:pave_y + img_height, pave_x:pave_x + img_width] = instance
            paved_instance = Image.fromarray(np.uint8(paved_instance)).convert('RGB')
        else:
            paved_instance = None

        if bodyPart is not None:
            paved_bodyPart = np.zeros((pave_h, pave_w, 3), dtype=int)
            paved_bodyPart[pave_y:pave_y + img_height, pave_x:pave_x + img_width] = bodyPart
            paved_bodyPart = Image.fromarray(np.uint8(paved_bodyPart)).convert('RGB')
        else:
            paved_bodyPart = None

        if coords.size(0) > 0:
            coords[:, [0, 2]] += pave_x
            coords[:, [1, 3]] += pave_y

            keep_idx_igs = ((coords[:, 2] - coords[:, 0]) >= 8) \
                           * ((coords[:, 3] - coords[:, 1]) >= 8) \
                           * (classes == 0)

            keep_idx_gts = ((coords[:, 2] - coords[:, 0]) >= limit) * \
                           (classes == 1)

            coords = coords[keep_idx_gts + keep_idx_igs, :]
            classes = classes[keep_idx_gts + keep_idx_igs]

            if instance_ids is not None:
                instance_ids = instance_ids[keep_idx_gts + keep_idx_igs]

        return paved_image, coords, classes, paved_instance, instance_ids, paved_bodyPart

    def __call__(self, image, classes, coords, instance=None, instance_ids=None, bodyPart=None):
        '''

        :param image: PIL Image
        :param classes: torch.Tensor
        :param coords: torch.Tensor
        :param instance: None or PIL Image
        :return: image [PIL Image], classes, coords
        '''

        image = ColorJitter(brightness=0.5)(image)

        if np.random.randint(0, 2) == 0:
            image, coords, instance, bodyPart = self.horizontal_flip(image=image, coords=coords, instance=instance,
                                                                     bodyPart=bodyPart)

        image, coords, classes, instance, instance_ids, bodyPart = self.augment_csp(image, coords, classes, instance,
                                                                                    instance_ids, bodyPart)

        return image, coords, classes, instance, instance_ids, bodyPart


if __name__ == '__main__':
    import os
    import importlib
    import numpy as np
    # np.random.seed(2)
    import matplotlib

    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    os.chdir('W:/1_projects/5_kia/software/ais_patrick/')

    from trainer.trainer_builder import TrainerBuilder
    from inferencer.utils import *

    def plot_gt_with_height(classes, coords, ax):

        distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231',
                           '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe',
                           '#008080', '#000080', '#aa6e28', '#fffac8', '#800000',
                           '#aaffc3', '#808000', '#ffd8b1', '#e6beff', '#808080',
                           '#FFFFFF', '#e6194b', '#3cb44b', '#ffe119', '#0082c8']

        labels = np.concatenate((np.expand_dims(classes, 1), coords), axis=1)

        coords = labels[:, -4:]
        for i in range(coords.shape[0]):
            xmin = coords[i, 0]
            ymin = coords[i, 1]
            xmax = coords[i, 2]
            ymax = coords[i, 3]
            w = xmax - xmin
            h = ymax - ymin

            ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
                                       color=distinct_colors[int(labels[i, 0])],
                                       fill=False, linewidth=1))

    trainer = TrainerBuilder() \
        .update(config_root='./cfg_train.yaml') \
        .create()
    # Load/ import models dynamically
    module = importlib.import_module('models.{}.{}.{}'.format(trainer.config.model[:-2],
                                                              trainer.config.model,
                                                              trainer.config.model))
    model = getattr(module, trainer.config.model.upper())
    model = model(trainer.config, trainer.device)
    net_config = model.get_net_config()
    trainer.update_config(new_config=net_config)
    trainer.prepare()

    mode = 'train'
    trainer.dataset_train.mode = mode
    trainer.dataset_train._augmentation = True

    n = 16
    fig, ax = get_subplots(num_subplots=n, img_res_h_w=trainer.config.img_res[mode])

    for i in range(min(n, len(trainer.dataset_train))):
        _, classes, coords, (image, _) = trainer.dataset_train.__getitem__(index=i)

        # print(coords)
        print(image.height, image.width)

        ax[i].imshow(image)
        ax[i].axis('off')
        plot_gt_with_height(classes=classes.numpy(), coords=coords.numpy(), ax=ax[i])

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()