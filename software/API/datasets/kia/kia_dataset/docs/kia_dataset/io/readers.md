[Back to Overview](../../README.md)



# kia_dataset.io.readers

> Readers for the data to make implementation of fixes and dataloaders easier.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer
* Philipp Heidenreich (Opel)


---
### *def* **read_image_png**(sequence_path, filename)

Read the png image as a numpy nd array.

* **sequence_path**: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
* **filename**: The filename of the actual frame without the ".png". This makes the name identical for all image based annotations.


---
### *def* **read_instance_mask**(sequence_path, filename)

Read the instance mask as a numpy nd array.

The returned mask is of shape h,w and contains the integer values for the instance ids. Conversion from RGB to int is already done.

* **sequence_path**: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
* **filename**: The filename of the actual frame without the ".png"/".exr". This makes the name identical for all image based annotations.


---
### *def* **read_depth**(sequence_path, filename)

Read the depth map as a numpy nd array.

The returned mask is of shape h,w and contains the float values for the depth in meters.

* **sequence_path**: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
* **filename**: The filename of the actual frame without the ".png"/".exr". This makes the name identical for all image based annotations.


---
### *def* **read_boxes_2d**(sequence_path, filename)

Read the 2d bounding boxes as a dict mapping ids to boxes.

The boxes returned are in the format the boxes have on the disk. No correction is done.

* **sequence_path**: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
* **filename**: The filename of the actual frame without the ".png"/".exr". This makes the name identical for all image based annotations.


---
### *def* **read_boxes_3d**(sequence_path, filename)

Read the 3d bounding boxes as a dict mapping ids to boxes.

The boxes returned are in the format the boxes have on the disk. No correction is done.

* **sequence_path**: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
* **filename**: The filename of the actual frame without the ".png"/".exr". This makes the name identical for all image based annotations.


