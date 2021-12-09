from typing import Tuple

from PIL import Image
import numpy as np

from patrick.datasets.base_dataset import BaseDataset


class WiderPersonDataset(BaseDataset):

    def __init__(self, config, image_set, mode, augmentation) -> None:
        super().__init__(config, image_set,  mode, augmentation)
        self._remove_bad_images()

    def _get_image_set(self):
        return self._image_set

    def _get_dataset_root_path(self) -> str:
        return self._config.wider_person_root

    def _get_class_names(self) -> list:
        # So far, ignore other classes
        return ['background', 'pedestrians', 'riders', 'partially_visible_persons', 'ignore_regions', 'crowd']

    def _load_image_ids(self) -> list:

        ids = []
        image_set_path = os.path.join(self._root_path, self._image_set, '{}.txt'.format(self._mode))
        with open(image_set_path) as file:
            ids += [line.strip() for line in file]

        return ids

    def _load_image_sizes(self) -> Tuple[list, list]:
        img_widths, img_heights = [], []
        for path in self._image_paths:
            image = Image.open(path, mode='r')
            img_widths.append(image.width)
            img_heights.append(image.height)
        return img_widths, img_heights

    def _load_image_paths(self) -> list:
        images_root_path = os.path.join(self._root_path, self._image_set, 'Images/')

        image_paths = []
        for single_id in self.image_ids:
            image_file_name = "{}.jpg".format(single_id)
            path = os.path.join(images_root_path, image_file_name)
            image_paths.append(path)
        return image_paths

    def _get_sequence_splits(self):
        return {
            'train': 'train',
            'test': 'test',
            'val': 'val',
            'mini': 'mini',
        }

    def _load_annotations(self) -> Tuple[list, list]:
        classes, coords = [], []
        annotation_root_path = os.path.join(self._root_path, self._image_set, 'Annotations/')
        classes_list = self.class_names

        for index, single_id in enumerate(self.image_ids):
            ann_path = os.path.join(annotation_root_path, '{}.jpg.txt'.format(single_id))

            with open(ann_path) as txt_file:

                _classes, _coords = [], []
                for n, line in enumerate(txt_file):

                    if n >= 1:
                        splits = line.split()
                        _classes += [int(splits[0])]
                        xmin = int(splits[1])
                        ymin = int(splits[2])
                        xmax = int(splits[3])
                        ymax = int(splits[4])
                        _coords.append([xmin, ymin, xmax, ymax])

            coords.append(np.array(_coords).astype(int))
            classes.append(np.array(_classes))

        return classes, coords

    def _remove_bad_images(self):
        for index in reversed(range(len(self.image_ids))):
            co = self.coordinates[index]
            if co.size == 0:
                self._delete_sample_at(index)
            else:
                box_widths = co[:, 2] - co[:, 0]
                box_heights = co[:, 3] - co[:, 1]
                if 0.0 in box_widths or 0.0 in box_heights:
                    self._delete_sample_at(index)

    def _delete_sample_at(self, index):
        del self.classes[index]
        del self.coordinates[index]
        del self._image_paths[index]
        del self.img_widths[index]
        del self.img_heights[index]
        del self.image_ids[index]

if __name__ == '__main__':
    from model.lib.utils import load_config, plot_label
    from datasets.fusion_dataset import FusionDataset
    import matplotlib.pyplot as plt
    import os

    os.chdir('W:/1_projects/5_kia/software/ais_patrick/')
    mode = 'mini'

    config, device = load_config()
    dataset = [WiderPersonDataset(config=config, image_set='', mode=mode, augmentation=False)]
    dataset = FusionDataset(config, dataset, augmentation=False)
    dataset.keep_classes(['background', 'pedestrians'])
    dataset.remove_gts(min_pixel_size_2d=0.01)
    print('Dataset: {} || Number of Images: {}'.format(config.dataset, len(dataset)))

    for i in range(len(dataset)):
    # for i in range(0, 20):
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




