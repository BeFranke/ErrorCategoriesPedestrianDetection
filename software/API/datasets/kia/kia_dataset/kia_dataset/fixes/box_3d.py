"""doc
# kia_dataset.fixes.box_3d

This is an executable script, that fixes the 3d bounding boxes in the kia dataset.

```bash
kia_fix_box_2d --data_path D:\\KIA-datasets\\extracted [--debug_mode true]
```

The script fixes the class attribute error and adds an estimated occlusion and depth field.
The corrected boxes are stored in '2d-bounding-box-fixed_json'.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer
* Philipp Heidenreich (Opel)
"""
import json
import numpy as np
import os
import argparse
from tqdm import tqdm
from pyquaternion import Quaternion

from kia_dataset.io.readers import *
from kia_dataset.io.helpers import *


__VERSION__ = '20201109'


# ****************
#     Main Fix
# ****************
def fix_bounding_boxes_3d(data_path, debug_mode=False):
    """
    Loads all boxes from all tranches and fixes the 3D bounding boxes.

    :param data_path: (str) An absolute path to the dataset, e.g. ('D:\\KIA-datasets\\extracted').
    """
    for path in get_sequence_paths(data_path):
        print(path)
        tranche, seq_number = get_seq_info(path)
        for filename in tqdm(get_box_3d_filenames(path, debug_mode)):   # tranche in [BIT_TRANCHE02, BIT_TRANCHE03, MAC_TRANCHE02]
            boxes = read_boxes_3d(path, filename)
            boxes = fix_class_id(tranche, boxes)
            boxes = fix_class_id_and_instance_id(tranche, boxes, path, seq_number)
            boxes = fix_center_and_size(tranche, boxes)
            write_boxes(path, filename, boxes)
        if debug_mode:
            print('')


# ****************
#      Fixes
# ****************
def fix_class_id(tranche, boxes):
    if tranche in [BIT_TRANCHE02]:
        for key in boxes:
            boxes[key]['class_id'] = boxes[key]['class']
            del boxes[key]['class']
    if tranche in [BIT_TRANCHE03]:
        for key in boxes:
            boxes[key]['class_id'] = boxes[key]['category']
            del boxes[key]['category']
    return boxes


def fix_class_id_and_instance_id(tranche, boxes, path, seq_number):
    fixed_boxes = boxes
    if tranche in [MAC_TRANCHE02]:
        # load instance-mapping_json
        path_instance_png = os.path.join(path, 'ground-truth', 'semantic-instance-segmentation_png')
        filename_mapping_json = os.path.join('_instance_mapping_' + seq_number + '.json')
        with open(os.path.join(path_instance_png, filename_mapping_json)) as f:
            instance_mapping = json.load(f)
        
        fixed_boxes = {}
        for key, box in boxes.items():
            if key in instance_mapping:
                (sR, sG, sB) = instance_mapping[key]['instance_color'].strip('()').split(',')
                box['instance_id'] = int(sR) * 2**16 + int(sG) * 2**8 + int(sB)
                box['class_id'] = instance_mapping[key]['class']
                fixed_boxes[key] = box
    return fixed_boxes


def fix_center_and_size(tranche, boxes):
    if tranche in [BIT_TRANCHE02]:
        for k, box in boxes.items():
            boxes[k] = convert_legacy_3d_box_format(box)
    return boxes


def convert_legacy_3d_box_format(box):
    x1 = np.array(box["x1"])
    x2 = np.array(box["x2"])
    x3 = np.array(box["x3"])
    x4 = np.array(box["x4"])
    x5 = np.array(box["x5"])
    x6 = np.array(box["x6"])
    x7 = np.array(box["x7"])
    x8 = np.array(box["x8"])
    # Remove from original box
    for i in range(8):
        del box["x{}".format(i+1)]
    
    # Compute scale dependent direction vectors
    FrontVec = x1-x5  # Vector defining the edge towards the front
    UpVec = x2-x1  # Vector defining the edge towards the top
    LeftVec = x3-x1  # Vector defining the edge towards the left

    # Use scale dependent direction vectors
    Length = magnitude_of_vec(FrontVec)
    Width = magnitude_of_vec(LeftVec)
    Height = magnitude_of_vec(UpVec)
    Center = np.mean([x1,x2,x3,x4,x5,x6,x7,x8], axis=0)

    # Assume Up to be z axis and then construct orientation from front vector.
    FrontVec = np.array([FrontVec[0], FrontVec[1], 0])
    FrontVec = np.array(normalize_vec(FrontVec))
    LeftVec = np.array([-FrontVec[1], FrontVec[0], 0])
    UpVec = np.array([0,0,1])

    # Rotation anno2world is an easy change of basis (assuming orthogonal directions)
    MatrixFromColumns = lambda a,b,c: np.stack((a,b,c), axis=-1)
    R_anno2world = MatrixFromColumns(FrontVec, LeftVec, UpVec)

    # Leading to
    box["size"] = [Length, Width, Height]
    box["center"] = list(Center)  # xyz
    box["rot"] = list(Quaternion(matrix=R_anno2world).elements)  # convert rotation matrix to quaternion elements
    return box


# ****************
#     Helpers
# ****************
def write_boxes(sequence_path, filename, boxes):
    if not os.path.exists(os.path.join(sequence_path, 'ground-truth', '3d-bounding-box-fixed_json')):
        os.mkdir(os.path.join(sequence_path, 'ground-truth', '3d-bounding-box-fixed_json'))
    # TODO once P5 process specified how a tool has to spcify what it did, add this to boxes object.
    json.dump(boxes, open(os.path.join(sequence_path, 'ground-truth', '3d-bounding-box-fixed_json', filename + '.json'), 'w'), indent=4)


# ****************
# Main Entry Point
# ****************
def main():
    parser = argparse.ArgumentParser(description='Fix the format of the bounding boxes and estimate occlusion and depth.')
    parser.add_argument('--data_path', type=str, required=True, help='The data path where the data was extracted.')
    parser.add_argument('--debug_mode', type=bool, default=False, help='True if you want to debug the script.')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print("The path does not exist: {}".format(args.data_path))

    fix_bounding_boxes_3d(args.data_path, args.debug_mode)

if __name__ == '__main__':
    main()
