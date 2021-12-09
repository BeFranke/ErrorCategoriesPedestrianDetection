[Back to Overview](../../README.md)



# kia_dataset.io.dfki_kia_sample_writer

> Write samples in the kia format.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer


---
---
## *class* **KIASampleWriter**(object)

Writes detection to a database.

* **output_dir**: The output directory where to write the results. The directory will be cleared before writing!
* **clear_output_folder**: If the output folder should be removed before writing. Default: True.
* **use_object_id_as_key**: If the key in the dictionary of annotations should be object id. When false, instance id as specified in E1.2.3 is used. Default: False.


---
### *def* **write**(*self*, detections_2d: List[KIADetection2D], detections_3d: List[KIADetection3D], sample_token: str, world_to_ego=None, world_mode=True) -> None

Write a list of samples for a frame.

* **detections_2d**: The 2d samples that should be written. (All must belong to the same frame and indices must align with 3d detections), when a value is None nothing is written (also no 3D).
* **detections_3d**: The 3d samples that should be written. (All must belong to the same frame and indices must align with 2d detections), when a value is None then only the 2D box is written.
* **sample_token**: The token for which the samples are (All must belong to the same frame).
* **world_to_ego**: Needs to be provided when detections are given in world coordinates.
* **world_mode**: Defines if the 3d box should be written only once in world coordinates. If false they are written in ego vehicle coordinates per camera. (Default: True).


---
### *def* **close**(*self*) -> None

Close the sample writer.


