import json
from typing import Tuple

from PIL import Image
import numpy as np

from patrick.datasets.base_dataset import BaseDataset


class ECPDataset(BaseDataset):
    # SOURCE: https://github.com/dbcollection/dbcollection/blob/a36f57a11bc2636992e26bba4406914162773dd9/dbcollection/datasets/caltech/caltech_pedestrian/detection.py#L337

    def __init__(self, config, image_set, mode, augmentation) -> None:
        super().__init__(config, image_set,  mode, augmentation)
        self._remove_bad_images()

    def _get_image_set(self):
        return self._image_set

    def _get_dataset_root_path(self) -> str:
        return self._config.ecp_root

    def _get_class_names(self) -> list:
        # So far, ignore other classes
        return ['background', 'pedestrian', 'person-group-far-away', 'bicycle-group', 'rider', 'scooter-group',
                'motorbike-group', 'rider+vehicle-group-far-away', 'buggy-group', 'tricycle-group', 'wheelchair-group',
                'bicycle', 'motorbike']

    def _load_image_ids(self) -> list:
        self.splits = self._get_sequence_splits()

        # ImageSets: 'day', 'night'
        inter_path = os.path.join(self._root_path,
                                '{}/img/{}/'.format(self._image_set, self.splits[self._mode]))

        subdirs = [x[0] for x in os.walk(inter_path)]
        subdirs.pop(0)

        ids = []
        for subdir in subdirs:
            id = os.listdir(subdir)
            city = subdir.split("/")[-1]
            id = [os.path.join(city, i) for i in id]
            ids += id

        # Remove '.png'
        ids_cut = [i[:-4] for i in ids]
        return ids_cut

    def _load_image_sizes(self) -> Tuple[list, list]:
        img_widths, img_heights = [], []
        for path in self._image_paths:
            image = Image.open(path, mode='r')
            img_widths.append(image.width)
            img_heights.append(image.height)
        return img_widths, img_heights

    def _load_image_paths(self) -> list:
        image_paths = []
        images_root_path = os.path.join(self._root_path,
                                        '{}/img/{}/'.format(self._image_set, self.splits[self._mode]))
        subdirs = [x[0] for x in os.walk(images_root_path)]
        subdirs.pop(0)

        for subdir in subdirs:
            id = os.listdir(subdir)
            path = [os.path.join(subdir, i) for i in id]
            image_paths += path
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
        annotation_root_path = os.path.join(self._root_path,
                                        '{}/labels/{}/'.format(self._image_set, self.splits[self._mode]))
        classes_list = self.class_names

        for index, single_id in enumerate(self.image_ids):
            ann_path = os.path.join(annotation_root_path, '{}.json'.format(single_id))

            with open(ann_path) as json_file:
                data = json.load(json_file)['children']

                _classes, _coords = [], []
                for obj in data:
                    _classes += [classes_list.index(obj['identity'])]

                    xmin = obj['x0']
                    ymin = obj['y0']
                    xmax = obj['x1']
                    ymax = obj['y1']
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

    os.chdir('W:/1_projects/5_kia/software/ssd/')
    mode = 'val'

    config, device = load_config()
    dataset = [ECPDataset(config=config, image_set='day', mode=mode, augmentation=False)]
    dataset = FusionDataset(config, dataset, augmentation=False)
    dataset.keep_classes(['background', 'pedestrian', 'rider'])
    dataset.merge_classes({'rider': ['pedestrian']})
    dataset.remove_gts(min_pixel_size_2d=0.0315)

    print('Dataset: {} || Number of Images: {}'.format(config.dataset, len(dataset)))

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




