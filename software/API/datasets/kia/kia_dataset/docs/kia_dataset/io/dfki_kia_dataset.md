[Back to Overview](../../README.md)



# kia_dataset.io.dfki_kia_dataset

> Prepare and read the KIA dataset.

## Preparation

It is assumed, that the zips/tars are downloaded into release and extracted to extracted. After extraction the autofix_data.py must be run on the data.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer


---
### *def* **warn_once**(message: str)

## Definitions

* `SEMANTIC_MAPPING`: Is a dictionary from class name to color in the semantic annotation.
* `KIADetection2D`: Is a namedtuple for 2d detections. It extends a CommonDetection by velocity, truncated, instance_id and 
* `KIADetection3D`: Is a namedtuple for 3d detections. It extends a CommonDetection by velocity, instance_id and object_id.




# Using the KIA-Dataset


---
---
## *class* **KIADataset**(Dataset)

The KIA Dataset is a synthetic dataset containing annotations for various tasks in automated driving.

You can use the dataset and directly access elements or use it as a pytorch dataloader during training.
```
InputType = NamedTuple("InputType", image=np.ndarray, scan=np.ndarray, projection=Projection)
OutputType = NamedTuple("OutputType", detections_3d=List[KIADetection3D], sample_token=str)

dataset = KIADataset(split, InputType, OutputType, data_path)
inp: InputType, outp: OutputType = dataset[42]

dataset.transformers.append(MyTransformer())

dataloader = dataset.to_pytorch(batch_size)
for inp, outp in dataloader:
preds = net(*inp)
loss = myLoss(preds, outp)
...
```

* **split**: Specifies for what the data is used train/val/test.
* **InputType**: The type definition (namedtuple) that the network input (features) should have.
* **OutputType**: The tye definition (namedtuple) that the network output (labels) should have.
* **data_path**: "/fakepath/KIA_dataset"` The path where the dataset is located.
* **data_sequences**: A list of sequences to use (e.g. kia_dataset.split.TRAIN_RELEASE_2).
* **data_use_world_coordinates**: If 3D annotations should be in world coordinates, when set to False the ego vehicle coordinate system will be used. (Default: True).
* **data_exr_images**: Controls if to use the exr image format for the RGB images. (Default: False)


---
### *def* **get_all_sample_tokens**(*self*, data_sequences: List[str]) -> List[str]

A function that computes the sample tokens assigned to a sequence and company.

* **data_sequences**: A list of sequences that are allowed.
* **returns**: A list of sample tokens that fulfill the requirements.


---
### *def* **get_sample_token**(*self*, sample_token: str) -> str

*(no documentation found)*

---
### *def* **get_sensor**(*self*, sample_token: str) -> str

*(no documentation found)*

---
### *def* **get_sequence**(*self*, sample_token: str) -> str

*(no documentation found)*

---
### *def* **get_frame**(*self*, sample_token: str) -> str

*(no documentation found)*

---
### *def* **get_world_token**(*self*, sample_token: str) -> str

*(no documentation found)*

---
### *def* **get_image**(*self*, sample_token: str) -> np.ndarray

Load the image from disk.

Loads the exr if self.exr_image is true, else it will load the png.

* **sample_token**: The sample token for which to load the image.
* **returns**: The image as a numpy array.


---
### *def* **get_depth**(*self*, sample_token: str) -> np.ndarray

Load the depth data from disk.

* **sample_token**: The sample token for which to load the depth.
* **returns**: The depth map as a numpy array.


---
### *def* **get_scan**(*self*, sample_token: str) -> np.ndarray

Load the lidar scan data from disk.

* **sample_token**: The sample token for which to load the depth.
* **returns**: The lidar scan poincloud as a numpy array. Shape is N,6 where N is the number of points, 0-3 is xyz and 4-6 is color info.
(likely to change in the future)


---
### *def* **get_semantic_segmentation**(*self*, sample_token: str) -> np.ndarray

Load the semantic segmentation data from disk.

* **sample_token**: The sample token for which to load the semantic segmentation.
* **returns**: The semantic segmentation is loaded as an image in np ndarray.


---
### *def* **get_instance_segmentation**(*self*, sample_token: str) -> np.ndarray

Load the instance segmentation data from disk.

* **sample_token**: The sample token for which to load the instance segmentation.
* **returns**: The instance segmentation is loaded as an image in np ndarray.


---
### *def* **get_detections_2d**(*self*, sample_token)

*(no documentation found)*

---
### *def* **convert_legacy_3d_box_format**(*self*, box)

*(no documentation found)*

---
### *def* **get_detections_3d**(*self*, sample_token)

*(no documentation found)*

---
### *def* **get_skeletons_2d**(*self*, sample_token)

*(no documentation found)*

---
### *def* **get_skeletons_3d**(*self*, sample_token)

*(no documentation found)*

---
### *def* **get_projection**(*self*, sample_token) -> Projection

*(no documentation found)*

---
### *def* **get_world_to_ego**(*self*, sample_token) -> Transform

*(no documentation found)*

---
### *def* **get_ego_to_world**(*self*, sample_token) -> Transform

*(no documentation found)*

