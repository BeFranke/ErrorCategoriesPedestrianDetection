import json
import os

import cv2
import numpy as np

# The dictionary of class name to the list of color values that belong to the
# class have been extracted from the class_list.json file within the Audi data.
# We merged all car and small vehicle instances as well as the truck and UVs.
class_colors = {
    "Car": ["#ff0000", "#c80000", "#960000", "#800000", "#00ff00", "#00c800",
            "#009600"],
    "Bicycle": ["#b65906", "#963204", "#5a1e01", "#5a1e1e"],
    "Pedestrian": ["#cc99ff", "#bd499b", "#ef59bf"],
    "Truck": ["#ff8000", "#c88000", "#968000", "#ffff00", "#ffffc8"],
}


def hex2rgb(h):
    """
    Convert a hex color string to a triplet of RGB integer values.

    :param h: The hex color string.

    :return: A triplet of integers between 0 and 255
    """
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def determine_2d_bbox(semantic_label, class_colors, min_area):
    """
    Takes a semantic image label and extracts 2D bounding boxes for the objects
    within the image. Only objects with a size bigger than the min_area will
    be extracted from the semantic label.

    :param semantic_label: The image that contains the colored semantic labels.
    :param class_colors: A dictionary of class name to a list of hex colors.
    :param min_area: The minimum area for which you want to generate a 2D bbox.

    :return: Dictionary of box id to bbox info and class information.
    """
    boxes = {}
    box_id = 0

    for key in class_colors.keys():
        for color in class_colors.get(key):

            # binary mask for each color (background = 0, foreground = 1)
            mask = cv2.inRange(semantic_label, hex2rgb(color), hex2rgb(color))
            # enumerated mask (background = 0, 1st connected component = 1, ...)
            _, mask_enum = cv2.connectedComponents(mask)

            for idx in range(np.max(mask_enum)):
                mask_idx = cv2.inRange(mask_enum, idx + 1, idx + 1)
                x0, y0, w, h = cv2.boundingRect(mask_idx)

                if w * h > min_area:
                    box_key = "box_" + str(box_id)
                    boxes[box_key] = {"class": key, "id": box_id,
                                      "2d_bbox": [x0, y0, x0 + w, y0 + h]}
                    box_id += 1

    return boxes


def generate_bboxes(sequences_folder):
    """
    Iterates through all the sequence folders within the given directory and
    generates a json file for every single semantic label image.

    :param sequences_folder: The folder that contains the sequence root folders.
    """

    # The smallest bounding box to be considered is equal to 1208//38 * 1920//38
    # because the smallest feature map of the SSD-300 model is of shape 38 x 38
    min_area = 32 * 51

    content = os.listdir(sequences_folder)
    dirs = [x for x in content if os.path.isdir(sequences_folder + x)]
    for dir in dirs:
        bboxes = generate_bboxes_for_sequence(sequences_folder + dir, min_area)
        save_bboxes(sequences_folder + dir, bboxes)
        print("Saved bboxes for: {}".format(sequences_folder + dir))


def save_bboxes(sequence_folder, bboxes):
    """
    Saves a dictionary of file_names and contents to the given sequence folder.

    :param sequence_folder: The folder where you want to store the json files.
    :param bboxes: Dict of file names matched to a dictionary of bbox truth.
    """
    label2d_path = os.path.join(sequence_folder, "label2D/cam_front_center/")
    if not os.path.isdir(label2d_path):
        os.makedirs(label2d_path)

    for name in bboxes.keys():
        path = os.path.join(label2d_path, name)
        with open(path, "w+") as json_file:
            json.dump(bboxes[name], json_file)


def generate_bboxes_for_sequence(directory, min_area):
    bboxes = {}
    semantic_labels_path = os.path.join(directory, "label/cam_front_center/")
    files = os.listdir(semantic_labels_path)
    for label_file in files:
        label_path = os.path.join(semantic_labels_path, label_file)
        semantic_label = cv2.imread(label_path)
        semantic_label = cv2.cvtColor(semantic_label, cv2.COLOR_BGR2RGB)
        bbox_file = label_file.replace("_label_", "_label2D_")
        bbox_file = bbox_file.replace(".png", ".json")
        bbox = determine_2d_bbox(semantic_label, class_colors, min_area)
        bboxes[bbox_file] = bbox
    return bboxes


if __name__ == '__main__':
    generate_bboxes("D:\datasets\A2D2\camera_lidar_semantic\\")
