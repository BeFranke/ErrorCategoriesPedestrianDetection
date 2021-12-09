"""doc
# kia_dataset.io.helpers

> Helpers for the dataset to make implementation of fixes and dataloaders easier.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer
* Philipp Heidenreich (Opel)
"""
from typing import List, Tuple
import random
import os


BIT_TRANCHE01 = 'b1'
BIT_TRANCHE02 = 'b2'
BIT_TRANCHE03 = 'b3'
BIT_TRANCHE04 = 'b4'
MAC_TRANCHE01 = 'm1'
MAC_TRANCHE02 = 'm2'
MAC_TRANCHE04 = 'm4'


def get_sequence_paths(root: str) -> List[str]:
    """
    Get all sequences that are in the root folder of the dataset.

    :param root: The path in which to search for sequences.
    """
    paths = []
    for item in os.listdir(root):
        if not os.path.isfile(os.path.join(root, item)) and not item.startswith("preview"):
            paths.append(os.path.join(root, item))
    paths = sorted(paths)
    return paths


def get_box_2d_filenames(sequence_path: str, debug_mode: bool = False) -> List[str]:
    """
    Get a list of all 2d bounding boxes in a sequence.

    :param sequence_path: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
    :param debug_mode: (Optional) Specifies if only a subset of one file should be returned. This is helpfull for debugging.
    """
    filenames = []
    path_2d = os.path.join(sequence_path, 'ground-truth', '2d-bounding-box_json')
    if os.path.exists(path_2d):
        filenames = os.listdir(path_2d)
        filenames = [f.replace('.json', '') for f in filenames]
        if debug_mode:
            filenames = [random.choice(filenames)]
    return filenames


def get_box_3d_filenames(sequence_path: str, debug_mode: bool = False) -> List[str]:
    """
    Get a list of all 3d bounding boxes in a sequence.

    :param sequence_path: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
    :param debug_mode: (Optional) Specifies if only a subset of one file should be returned. This is helpfull for debugging.
    """
    filenames = []
    path_3d = os.path.join(sequence_path, 'ground-truth', '3d-bounding-box_json')
    if os.path.exists(path_3d):
        filenames = os.listdir(path_3d)
        filenames = [f.replace('.json', '') for f in filenames]
        if debug_mode:
            filenames = [random.choice(filenames)]
    return filenames


def get_seq_info(sequence_path: str) -> Tuple[str, str]:
    """
    Get information about the sequence.

    :param sequence_path: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
    :return: A tuple containing the Tranche and the sequence number (as string).
    """

    if sequence_path.split('/')[-1] == 'mini' or sequence_path.split('/')[-1] == 'inference':
        fake_seq = '0001'
        return BIT_TRANCHE01, fake_seq

    seq_producer = os.path.basename(sequence_path).split('_')[0]
    seq_number = os.path.basename(sequence_path).split('_')[3].split('-')[0]
    if (seq_producer == 'bit' and int(seq_number) in range(1, 25)):
        return BIT_TRANCHE01, seq_number
    if (seq_producer == 'bit' and int(seq_number) in range(25, 70)):
        return BIT_TRANCHE02, seq_number
    if (seq_producer == 'bit' and int(seq_number) in range(70, 128)):
        return BIT_TRANCHE03, seq_number
    if (seq_producer == 'bit' and int(seq_number) in range(147, 217)):
        return BIT_TRANCHE04, seq_number
    if (seq_producer == 'mv' and int(seq_number) in range(1, 14)):
        return MAC_TRANCHE01, seq_number
    if (seq_producer == 'mv' and int(seq_number) in range(27, 40)):
        return MAC_TRANCHE02, seq_number
    if (seq_producer == 'mv' and int(seq_number) in range(40, 56)):
        return MAC_TRANCHE04, seq_number

    raise RuntimeError('Unsupported data tranche! Producer: {}, Sequence: {}'.format(seq_producer, seq_number))
