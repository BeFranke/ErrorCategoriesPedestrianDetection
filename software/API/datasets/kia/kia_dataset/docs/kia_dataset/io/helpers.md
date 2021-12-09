[Back to Overview](../../README.md)



# kia_dataset.io.helpers

> Helpers for the dataset to make implementation of fixes and dataloaders easier.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer
* Philipp Heidenreich (Opel)


---
### *def* **get_sequence_paths**(root: str) -> List[str]

Get all sequences that are in the root folder of the dataset.

* **root**: The path in which to search for sequences.


---
### *def* **get_box_2d_filenames**(sequence_path: str, debug_mode: bool = False) -> List[str]

Get a list of all 2d bounding boxes in a sequence.

* **sequence_path**: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
* **debug_mode**: (Optional) Specifies if only a subset of one file should be returned. This is helpfull for debugging.


---
### *def* **get_box_3d_filenames**(sequence_path: str, debug_mode: bool = False) -> List[str]

Get a list of all 3d bounding boxes in a sequence.

* **sequence_path**: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
* **debug_mode**: (Optional) Specifies if only a subset of one file should be returned. This is helpfull for debugging.


---
### *def* **get_seq_info**(sequence_path: str) -> Tuple[str, str]

Get information about the sequence.

* **sequence_path**: The path to the folder of the sequence. It has subfolders sensor and ground-truth.
* **returns**: A tuple containing the Tranche and the sequence number (as string).


