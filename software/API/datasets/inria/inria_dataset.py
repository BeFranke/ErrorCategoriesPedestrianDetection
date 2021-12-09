import json
import os

from typing import Tuple

from PIL import Image
import numpy as np

from patrick.datasets.base_dataset import BaseDataset


class INRIADataset(BaseDataset):
    # SOURCE: https://github.com/dbcollection/dbcollection/blob/a36f57a11bc2636992e26bba4406914162773dd9/dbcollection/datasets/caltech/caltech_pedestrian/detection.py#L337

    def __init__(self, config, image_set, mode, augmentation) -> None:
        super().__init__(config, image_set,  mode, augmentation)

    def _get_image_set(self):
        return self.image_set

    def _get_dataset_root_path(self) -> str:
        return self.config.inria_root

    def _get_class_names(self) -> list:
        return ['ignore', 'person']

    def _load_image_ids(self) -> list:
        # Only 'V000' directory is filled with images of persons - 'V001' is garbage!
        self.splits = self._get_sequence_splits()
        set_path = os.path.join(self.root_path, self.image_set,
                                '{}/V000/images/'.format(self.splits[self.mode]))
        ids = os.listdir(set_path)
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
        images_root_path = os.path.join(self.root_path, self.image_set,
                                        '{}/V000/images/'.format(self.splits[self.mode]))
        for single_id in self.image_ids:
            path = os.path.join(images_root_path, '{}.png'.format(single_id))
            image_paths.append(path)
        return image_paths

    def _get_sequence_splits(self):
        return {
            'train': 'set00',
            'val': 'set01',
            'mini': 'set02',
        }

    def _load_annotations(self) -> Tuple[list, list]:
        classes, coords = [], []
        annotation_root_path = os.path.join(self.root_path, self.image_set,
                                            '{}/V000/annotations/'.format(self.splits[self.mode]))
        classes_list = self.class_names

        for index, single_id in enumerate(self.image_ids):
            ann_path = os.path.join(annotation_root_path, '{}.json'.format(single_id))

            with open(ann_path) as json_file:
                data = json.load(json_file)

                _classes, _coords = [], []
                for obj in data:
                    _classes += [classes_list.index(obj['lbl'])]

                    # 'pos' in INRIA is in [xmin,ymin,w,h] - We need [xmin,ymin,xmax,ymax]
                    xmin = obj['pos'][0]
                    ymin = obj['pos'][1]
                    xmax = obj['pos'][0] + obj['pos'][2]
                    ymax = obj['pos'][1] + obj['pos'][3]
                    _coords.append([xmin, ymin, xmax, ymax])

                coords.append(np.array(_coords).astype(int))
                classes.append(np.array(_classes))

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

if __name__ == '__main__':
    from model.lib.utils import load_config, plot_label
    import matplotlib.pyplot as plt
    import os

    os.chdir('W:/1_projects/5_kia/software/ssd/')
    mode = 'test'

    config, device = load_config()
    dataset = INRIADataset(config=config, image_set='extracted_data', mode=mode, augmentation=False)

    # for i in range(len(dataset)):
    for i in range(0, 20):
        image, classes, coords, image_path = dataset.__getitem__(i)
        image = Image.open(image_path, mode='r').convert('RGB')

        dpi, scale = 80, 0.5
        width, height = image.width, image.height
        figsize = scale * width / float(dpi), scale * height / float(dpi)

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(image)
        ax.axis('off')
        ax.axis('tight')

        labels = np.concatenate((np.expand_dims(classes, axis=1), coords), axis=1)
        plot_label(labels=labels, ax= ax, img_size=(image.width, image.height))

    plt.show()




