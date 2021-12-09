"""
This script is supposed to find corrupted image ids within a given COCO
dataset. We found that some images might have no detections at all or they might
have detections that have a zero height or width.
"""

import os

from pycocotools.coco import COCO


def filter_no_anns(ann_file):
    """
    Finds all the train2017 in the given COCO annotation file that have no
    annotations/detections. Since this might break the detector during training
    we need to remove these train2017. The result will be a list of str ids of
    the train2017 that have no annotations.

    :param ann_file: The complete path to the annotations file.

    :return: A list of string ids that are bad.
    """
    bad_images = []
    coco_annotations = COCO(os.path.join(ann_file))
    image_ids = list(sorted(coco_annotations.imgs.keys()))
    for single_id in image_ids:
        annotations_ids = coco_annotations.getAnnIds(imgIds=single_id)
        if not annotations_ids:
            bad_images.append(str(single_id))
    return bad_images


def filter_zero_area(ann_file):
    """
    During training we discovered that some train2017 within the COCO datasets
    have detections that either have a zero height or width. This is problematic
    during training and validation because the Location loss function within the
    SSD detector uses the logarithm to determine the width and height offset
    loss. If the height or the width is zero the function evaluates to infinity.

    :param ann_file: The complete path to the annotations file.

    :return: A list of string ids that are bad.
    """
    bad_images = []
    coco_annotations = COCO(os.path.join(ann_file))
    image_ids = list(sorted(coco_annotations.imgs.keys()))
    for single_id in image_ids:
        annotations_ids = coco_annotations.getAnnIds(imgIds=single_id)
        targets = coco_annotations.loadAnns(annotations_ids)
        positive_area = [t['area'] == 0 for t in targets]
        if True in positive_area:
            bad_images.append(str(single_id))
    return bad_images


def create_blacklist(annotation_path, coco_image_set):
    """
    Creates a txt file with image ids of the given coco image set that can be
    found in the annotations path. Some train2017 might be coruppted so we need
    to remove them before we start training. Saves the blacklist.txt file in the
    same folder as this script.

    :param annotation_path: The path where this function expects to find the
                            COCO annotation files.
    :param coco_image_set: The blacklist will be generated for this image set.
    """
    output_file_path = "./img_ids_blacklist_{}.txt".format(coco_image_set)
    ann_file_name = "instances_{}.json".format(coco_image_set)
    ann_file_path = os.path.join(annotation_path, ann_file_name)

    missing_ann_ids = filter_no_anns(ann_file_path)
    zero_area_ids = filter_zero_area(ann_file_path)
    bad_images = missing_ann_ids + zero_area_ids

    with open(output_file_path, "w+") as file:
        blacklist_string = ','.join(bad_images)
        file.write(blacklist_string)
        file.close()


# Sample usage of the blacklist function above
create_blacklist("2017/annotations/", "train2017")
