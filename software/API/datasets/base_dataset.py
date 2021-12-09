from copy import deepcopy
from typing import Tuple, List

import numpy as np
import torchvision.transforms.functional as FT
import matplotlib.pyplot as plt

import torch
from PIL import Image
from torch.utils import data

from patrick.datasets.utils import *
from patrick.models.lib.utils import calc_iou_tensor
from patrick.datasets.augmentation import CSPAugmentation
from benedikt import debug


class BaseDataset(data.Dataset):
    """
    This dataset class serves as a base class for any dataset that should be fed
    into the SSD model. You simply need to implement the following functions to
    load the dataset in a format that is compatible with our training, inference
    and evaluation scripts:

    Unimplemented methods:
        - _get_dataset_root_path
        - _get_class_names
        - _load_image_ids
        - _load_image_sizes
        - _load_image_paths
        - _load_annotations
    """

    def __init__(self, config, image_set, mode, augmentation) -> None:
        super().__init__()

        self.mode = mode
        self.config = config
        self.image_set = image_set
        self._augmentation = augmentation

        self.augmentation = CSPAugmentation(config=self.config)

        self.class_names = self._get_class_names()
        self.root_path = self._get_dataset_root_path()
        self.image_ids = self._load_image_ids()
        self.image_paths = self._load_image_paths()
        self.img_widths, self.img_heights = self._load_image_sizes()
        self.classes, self.coordinates = self._load_annotations()

    def _get_image_set(self):
        """
        Retrieves the string name of the current image set.
        """
        raise NotImplementedError

    def _get_dataset_root_path(self) -> str:
        """
        Returns the path to the root folder for this dataset. For PascalVOC this
        would be the path to the VOCdevkit folder.
        """
        raise NotImplementedError

    def _get_class_names(self) -> List[str]:
        """
        Returns the list of class names for the given dataset.
        """
        raise NotImplementedError

    def _load_image_ids(self) -> List:
        """
        Returns a list of strings with image ids that are part of the dataset.
        The image ids usually indicated the image file name.
        """
        raise NotImplementedError

    def _load_image_sizes(self) -> Tuple[List[int], List[int]]:
        """
        Retrieves the width and height of all images in the given dataset and
        returns two lists with all integer widths and all integer heights.
        """
        raise NotImplementedError

    def _load_image_paths(self) -> List[str]:
        """
        Returns a list of file paths for each of the images.
        """
        raise NotImplementedError

    def _load_annotations(self) -> Tuple[List, List]:
        """
        Loads the categories for each detection for each image and loads the
        bounding box information for each detection for each image.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns the number of images within this dataset.
        """
        return len(self.image_ids)

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
        image = Image.open(image_path, mode='r').convert('RGB')

        classes = deepcopy(self.classes[index])
        coords = deepcopy(self.coordinates[index])
        classes = torch.tensor(classes, dtype=torch.long)
        coords = torch.tensor(coords, dtype=torch.float32)

        image, target = self.get_target(image=image, classes=classes, coords=coords, image_path=image_path, index=index)

        # Tensor of Input Image
        image_tensor = FT.to_tensor(image)
        image_tensor = FT.normalize(image_tensor, mean=self.config.norm_mean, std=self.config.norm_std)

        return image_tensor, target

    def get_target(self, image, classes, coords, image_path, index):
        raise NotImplementedError

    def calc_basic_stats(self):
        self.stats = {'Basic': {'NumImages': len(self.image_ids),
                                'NumGroundtruths': self.get_num_gts()}
                      }

    def get_num_gts(self):
        num_igs, num_gts = 0, 0

        for classes in self.classes:
            if classes.size > 0:
                num_gts += (classes == 1).sum()
                num_igs += (classes == 0).sum()

        return {'person': int(num_gts), 'ignore': int(num_igs), 'total': int(num_gts+num_igs)}

    # Responsible for batching and padding inputs
    @staticmethod
    def collate_fn(samples_in_batch):
        """
        Helps to create a real batch of inputs and targets. The function
        receives a list of single samples and combines the inputs as well as the
        targets to single lists.

        :param samples_in_batch: A list of quadruples of an image, its class,
            its coordinates and file path.

        :return: A batch of samples.
        """
        images = [sample[0] for sample in samples_in_batch]
        images = torch.stack(images, dim=0)

        keys = list(samples_in_batch[0][1])
        target = {}
        for key in keys:
            target[key] = [sample[1][key] for sample in samples_in_batch]

        return images, target

    def remove_classes(self, remove):
        """
        Removes the classes wih the given names from this dataset. This function
        will adjust the classes and coordinates tensors of the dataset. If an
        image has zero detections after removing a specific class from the
        dataset, the image will be removed from the dataset.

        :param remove: A list of class names that should be removed from
            the dataset.
        """
        remove_classes = [self.class_names.index(x) for x in remove]
        keep = [x for x in self.class_names if x not in remove]
        reindex = {self.class_names.index(x): i for i, x in enumerate(keep)}
        self.class_names = keep
        for index in reversed(range(self.__len__())):
            class_array = self.classes[index]
            delete_mask = np.isin(class_array, remove_classes)
            class_array = np.array(class_array)[~delete_mask]
            if class_array.size == 0:
                del self.classes[index]
                del self.coordinates[index]
                del self.image_paths[index]
                del self.image_ids[index]
                del self.img_widths[index]
                del self.img_heights[index]
            else:
                class_array = [reindex[id] for id in class_array]
                self.classes[index] = np.array(class_array)
                coords_array = self.coordinates[index][~delete_mask]
                self.coordinates[index] = coords_array

    def keep_classes(self, keep_names):
        """
        Removes all classes from this dataset except for the class with the
        given names.

        :param keep_names: A list of class names that should be kept.
        """
        remove_names = [x for x in self.class_names if x not in keep_names]
        self.remove_classes(remove_names)

    def merge_classes(self, merge_dict):
        """
        Takes in a dictionary with a key class name and a list of class names as
        value that should be merged together. If the key is "car" and the value
        is ["truck", "van"] the result of this function will be that
        there is no more class truck and van but the classes have been merged to
        only car including the car class.

        **NOTE:** Please make sure that the key is not part of the list and
        that no class is mentioned twice.

        :param merge_dict: The dictionary which allows you to specify the merge
            instructions.
        """
        reindex = {}
        remove = [x for sublist in merge_dict.values() for x in sublist]
        keep = [x for x in self.class_names if x not in remove]

        for i, name in enumerate(self.class_names):
            if name in remove:
                new_index = self.find_parent_merger(name, merge_dict, keep)
            else:
                new_index = keep.index(name)
            reindex[i] = new_index

        for index in range(self.__len__()):
            class_array = self.classes[index]
            class_array = np.array([reindex[id] for id in class_array])
            self.classes[index] = class_array

        self.class_names = keep

    def find_parent_merger(self, name, merge_dict, keep):
        """
        Returns the index of key within the dictionary whose value contains the
        name.
        """
        for i, x in enumerate(merge_dict.values()):
            if name in x:
                return keep.index(list(merge_dict.keys())[i])
        return 0

    def rename_class(self, old_name, new_name):
        """
        Renames the class with the given old name to the given new name.
        """
        if old_name not in self.class_names:
            raise AttributeError("The old name does not exist.")
        if new_name in self.class_names:
            raise AttributeError("The new name already exists.")
        self.class_names = [new_name if old_name == name else name for name in self.class_names ]

    def remove_bad_images(self):
        '''
        Remove all Images with no coords or Groundtruths
        '''
        for index in reversed(range(len(self.image_ids))):
            co = self.coordinates[index]
            cl = self.classes[index]

            if co.size == 0 or cl.sum() == 0:
                self._delete_sample_at(index)
                continue

        self.stats['RemoveBadImages'] = {'NumImages': len(self.image_ids),
                                         'NumGroundtruths': self.get_num_gts()}

    def print_stats(self, dataset):
        print('----------------------------------------------------------------------------------------')
        print('{}-Dataset: {} | ImageSet: {}'
              .format(dataset[1].upper(), dataset[0].upper(), self.image_set.upper()))

        for type, value in self.stats.items():
            print('{:^16} - Images: {} - GTsTotal: {} - GTsPerson: {} - GTsIgnore: {}'
                  .format(type,
                          value['NumImages'],
                          value['NumGroundtruths']['total'],
                          value['NumGroundtruths']['person'],
                          value['NumGroundtruths']['ignore'])
                  )
        print('----------------------------------------------------------------------------------------')

    def _delete_sample_at(self, index):
        del self.classes[index]
        del self.coordinates[index]
        del self.image_paths[index]
        del self.img_widths[index]
        del self.img_heights[index]
        del self.image_ids[index]
        del self.occlRatio[index]

    def extract_reasonable_height_and_width(self, height=50, width=10):
        '''
        Definition of 'reasonable' is based on the Height of the Bounding Box. Just switch class from 1 to 0
        '''
        for index in range(len(self.image_ids)):

            coords = self.coordinates[index]
            if coords.size == 0: continue

            heights = coords[:, 3] - coords[:, 1]
            widths = coords[:, 2] - coords[:, 0]

            if width is not None:
                keep_index = np.logical_and((heights >= height),
                                            np.logical_and((widths >= width), (self.classes[index] == 1)))
            else:
                keep_index = np.logical_and((heights >= height), (self.classes[index] == 1))

            self.classes[index] *= keep_index.astype('int')

        self.stats['ReasonableHeightWidth'] = {'NumImages': len(self.image_ids),
                                               'NumGroundtruths': self.get_num_gts()}

    def make_ignore(self):
        for index in range(len(self.image_ids)):
            keep_index = self.classes[index] == 1
            self.classes[index] *= keep_index

        self.stats['MakeIgnore'] = {'NumImages': len(self.image_ids),
                                    'NumGroundtruths': self.get_num_gts()}

    def extract_reasonable_occlusion(self, occlusion_max=0.5):
        for index in reversed(range(len(self.image_ids))):
            occlusions = self.occlusion[index]

            keep_index = (occlusions <= occlusion_max) * (self.classes[index] == 1)
            self.classes[index] *= keep_index

        # It is possible that no true GT survives!
        self.remove_bad_images()

        self.stats['ReasonableOccl'] = {'NumImages': len(self.image_ids),
                                        'NumGroundtruths': self.get_num_gts()}

    def suppress_overlaps(self, iou=0.9):
        '''
        Combine Groundtruths with high overlaps. Keep surrounding Box of 2 suppressed boxes
        Example: Cityscapes - 2 Annotations for nearly the same (Pedestrian and Rider)
        '''

        num_overlaps = 0
        for index in range(len(self.image_ids)):

            for c in range(len(self.class_names)):
                if c not in self.classes[index].tolist(): continue

                _num_overlaps = 1
                while _num_overlaps >= 1:

                    # Only coords for classes 1
                    # Store ignore_area (0) coords and classes
                    idx = self.classes[index] == c
                    classes = self.classes[index][idx]
                    coords = self.coordinates[index][idx, :]

                    # Remove whole class from dataset attributes and add them after each Overlap Suppression again
                    self.classes[index] = self.classes[index][~idx]
                    self.coordinates[index] = self.coordinates[index][~idx, :]

                    coords = torch.from_numpy(coords).type(torch.float32)
                    ious = calc_iou_tensor(box1=coords, box2=coords).numpy() * np.tri(N=coords.size(0), k=-1)

                    if np.isnan(np.min(ious)):
                        print('Dataset Issue: Loaded faulty coordinates!')

                    # Get pairs of suppressed boxes
                    suppress_idx = np.where(ious > iou)
                    _num_overlaps = suppress_idx[0].size
                    num_overlaps += _num_overlaps

                    overlaps = np.concatenate(suppress_idx, axis=0)
                    # Check for multiple overlaps - Can occur if we have a Triple Occlusion - Just use a simple
                    # while loop do get the best out of it!
                    multiple_overlaps = overlaps.size != np.unique(overlaps).size

                    if multiple_overlaps:
                        i = [suppress_idx[0].tolist()[0]]
                        j = [suppress_idx[1].tolist()[1]]
                    else:
                        i = suppress_idx[0].tolist()
                        j = suppress_idx[1].tolist()

                    # Add all unrelated coordinates to new coordinates
                    coords_new, classes_new = [], []
                    for n in range(coords.size(0)):
                        if n not in overlaps.tolist():
                            coords_new.append(np.array(coords[n, :]))
                            classes_new.append(c)

                    if num_overlaps != 0:
                        for _i, _j in zip(i, j):
                            box_1 = coords[_i]
                            box_2 = coords[_j]

                            box = np.array([np.min((box_1[0], box_2[0])),
                                            np.min((box_1[1], box_2[1])),
                                            np.max((box_1[2], box_2[2])),
                                            np.max((box_1[3], box_2[3]))])
                            coords_new.append(box)
                            classes_new.append(c)

                    _coords = np.concatenate([self.coordinates[index], np.array(coords_new)])
                    self.coordinates[index] = _coords
                    _classes = np.concatenate([self.classes[index], np.array(classes_new)])
                    self.classes[index] = _classes

        self.stats['SuppressOverlaps'] = {'NumImages': len(self.image_ids),
                                          'NumGroundtruths': self.get_num_gts()}
