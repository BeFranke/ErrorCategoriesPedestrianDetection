"""doc
# kia_dataset.fixes.box2d

This is an executable script, that fixes the 2d bounding boxes in the kia dataset.

```bash
kia_fix_box_2d --data_path D:\\KIA-datasets\\extracted [--debug_mode true]
```

The script fixes the class attribute error and adds an estimated occlusion and depth field.
The corrected boxes are stored in '2d-bounding-box-fixed_json'.

## Authors and Contributors
* Philipp Heidenreich (Opel), Lead-Developer
* Michael Fürst (DFKI)
"""
import cv2
import json
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from kia_dataset.io.readers import *
from kia_dataset.io.helpers import *


__VERSION__ = '20210321'


# ****************
#     Main Fix
# ****************
def fix_bounding_boxes_2d(data_path, debug_mode=False, estimate_depth=True):
    """
    Loads all boxes from all tranches and fixes the 2D bounding boxes.

    :param data_path: (str) An absolute path to the dataset, e.g. ('D:\\KIA-datasets\\extracted').
    """
    for path in get_sequence_paths(data_path):
        print(path)
        tranche, seq_number = get_seq_info(path)
        for filename in tqdm(get_box_2d_filenames(path, debug_mode)):   # tranche in [BIT_TRANCHE02, BIT_TRANCHE03, MAC_TRANCHE02]

            # if os.path.exists(os.path.join(path, 'ground-truth', '2d-bounding-box-fixed_json')): continue
            boxes = read_boxes_2d(path, filename)
            boxes = fix_class_id(tranche, boxes)
            boxes = fix_class_id_and_instance_id(tranche, boxes, path, seq_number)
            boxes = post_proc_estimate_occlusion_and_depth(tranche, boxes, path, filename, estimate_depth)
            if debug_mode:
                image = read_image_png(path, filename)
                visualize_boxes(boxes, image)
            else:
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


def post_proc_estimate_occlusion_and_depth(tranche, boxes, path, filename, estimate_depth):
    if tranche in [BIT_TRANCHE04]:
        instance = read_instance_mask(path, filename)
        if estimate_depth:
            depth = read_depth(path, filename)
        for instance_id, box in boxes.items():
            mask = instance == int(instance_id)     # [H, W]
            instance_pixels = mask.sum()
            # As BIT has inaccurate boxes in T2 and T3 use a bigger correction factor
            correction_factor = 2.5 if tranche in [BIT_TRANCHE02, BIT_TRANCHE03] else 2.0
            # approximate occlusion from bounding box size and number of pixels per instance mask
            occlusion = 1.0 - correction_factor * instance_pixels / (box['w'] * box['h'])
            occlusion = np.clip(occlusion, 0.0, 1.0)
            # convert from numpy objects to numeric objects for JSON serialization
            box['instance_pixels'] = int(instance_pixels)
            box['occlusion_est'] = round(float(occlusion), 2)

            # Get visible Bounding Box - CityPersons comes with in [xmin, ymin, w, h] Format!
            xmin = np.argmax(mask.max(axis=0))
            ymin = np.argmax(mask.max(axis=1))
            W, H = mask.shape[1], mask.shape[0]
            xmax =  W - np.argmax(np.flip(mask.max(axis=0))) - 1
            ymax = H - np.argmax(np.flip(mask.max(axis=1))) - 1
            box['vis_bbox'] = [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)]
            box['vis_ratio_bb'] = np.round( (box['vis_bbox'][2] * box['vis_bbox'][3]) / (box['w'] * box['h']), 4)
            box['occl_ratio_bb'] = np.round(1.0 - box['vis_ratio_bb'], 4)
            box['bbox_area'] = int(box['w'] * box['h'])
            box['bbox_cityscapes'] = [float(box['c_x'] - box['w'] / 2.0),   # xmin
                                      float(box['c_y'] - box['h'] / 2.0),   # ymin
                                      float(box['w']),                      # w
                                      float(box['h']),                      # h
                                      ]

            if estimate_depth:
                instance_depth = np.median(depth[mask]) if instance_pixels > 0 else -1.0
                box['depth'] = round(float(instance_depth), 2)
    return boxes


# ****************
#     Helpers
# ****************
def write_boxes(sequence_path, filename, boxes):
    if not os.path.exists(os.path.join(sequence_path, 'ground-truth', '2d-bounding-box-fixed_json')):
        os.mkdir(os.path.join(sequence_path, 'ground-truth', '2d-bounding-box-fixed_json'))
    # TODO once P5 process specified how a tool has to spcify what it did, add this to boxes object.
    json.dump(boxes, open(os.path.join(sequence_path, 'ground-truth', '2d-bounding-box-fixed_json', filename + '.json'), 'w'), indent=4)


# ****************
#  Visualizations
# ****************
def visualize_boxes(boxes, image):
    print('')
    for key in boxes:
        c_x = int(boxes[key]['c_x'])
        c_y = int(boxes[key]['c_y'])
        w = int(boxes[key]['w'])
        h = int(boxes[key]['h'])
        instance_id = int(boxes[key]['instance_id'])
        instance_pixels = int(boxes[key]['instance_pixels'])
        depth = boxes[key]['depth']
        occlusion = boxes[key]['occlusion_est']
        # print boxes
        data = (instance_id, c_x, c_y, w, h, instance_pixels, depth, occlusion)
        print('instance_id: %8d, c_x: %4d, c_y: %4d, w: %4d, h: %4d, pixels: %6d, depth: %5.1f, occl: %4.2f' % data)
        # overlay boxes
        pt1 = (c_x - w // 2, c_y - h // 2)
        pt2 = (c_x + w // 2, c_y + h // 2)
        if occlusion >= 0.6:
            color = (0, 0, 255)     # red for heavy occlusion
        elif occlusion >= 0.4:
            color = (0, 255, 255)   # yellow for medium occlusion
        elif occlusion >= 0.2:
            color = (0, 255, 0)     # green for weak occlusion  
        else:
            color = (255, 255, 0)   # cyan for no occlusion
        if instance_pixels > 0:
            cv2.rectangle(image, pt1, pt2, color, 2)
            cv2.putText(image, '%d' % instance_id, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(image, '%3.1f' % depth, pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.imshow('Fixed Boxes', image)
    while not cv2.waitKey() == 113:
        pass


# ****************
# Main Entry Point
# ****************
def main():

    parser = argparse.ArgumentParser(description='Fix the format of the bounding boxes and estimate occlusion and depth.')
    parser.add_argument('--data_path', type=str, required=False, help='The data path where the data was extracted.',
                        default='/home/patrick/kia/input/datasets/kia/')
    parser.add_argument('--debug_mode', type=bool, default=False, help='True if you want to debug the script. This does not write but only visualize.')
    parser.add_argument('--estimate_depth', type=bool, default=False, help='True if you want to do the expensive depth estimation.')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print("The path does not exist: {}".format(args.data_path))

    fix_bounding_boxes_2d(args.data_path, args.debug_mode, args.estimate_depth)

if __name__ == '__main__':
    main()
