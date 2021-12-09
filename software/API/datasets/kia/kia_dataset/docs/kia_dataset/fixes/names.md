[Back to Overview](../../README.md)



# kia_dataset.fixes.names

> Fix the names of the files to adhere to the filename schema.

## Installation/Setup

Follow the instructions in the main readme.

## Running

Follow the instructions in the main readme. This implements the following commands:
* `kia_fix_names_tranche_1`
* `kia_fix_names_tranche_2`

Parameters:
* `--data_path`: The path where the extracted data is be stored.


---
### *def* **clean_emtpy_folders**(data_path)

Remove all empty folders from the data.

* **data_path**: (str) The path where the extracted data is.


---
### *def* **fix_sequence_folder_prefix_BIT_Tranche1**(data_path)

Add the bit prefix to the data of tranche 1 from bit.

* **data_path**: (str) The path where the extracted data is.


---
### *def* **fix_filename_scheme_MV_Tranche1**(data_path)

Rename the ground truth in MV tranche 1 from "ground_truth" to "ground-truth"

* **data_path**: (str) The path where the extracted data is.


---
### *def* **fix_filename_scheme_BIT_Tranche2**(data_path)

In tranche 2 of BIT fix the prefix for global annotations to "world-" from "sequence_".

* **data_path**: (str) The path where the extracted data is.


---
### *def* **tranche_1**()

Entry point for kia_fix_names_tranche_1.


---
### *def* **tranche_2**()

Entry point for kia_fix_names_tranche_2.


