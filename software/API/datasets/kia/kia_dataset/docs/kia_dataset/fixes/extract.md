[Back to Overview](../../README.md)



# kia_dataset.fixes.extract

> Extracts all remaining archive files and moves bit tranche 1 files to correct directory

## Installation/Setup

Follow the instructions in the main readme.

## Running

Follow the instructions in the main readme. This implements the following commands:
* `kia_fix_extract`

Parameters:
* `--data_path`: The path where the extracted data is be stored.
* `--no_delete`: When set, archive files will not be deleted after files have been extracted.
* `--dry_run`: No data will be modified. Just to check what will be fixed.
* `--logfile`: When a file is specified, outputs are logged into the given file.


---
### *def* **find_archive_files**(data_path: str) -> list

Finds all remaining archive files in the dataset path.

* **data_path**: (str) An absolute path to the dataset, e.g. ('D:\\KIA-datasets\\extracted').
* **returns**: (list) A list of all archive files in dataset path.


---
### *def* **extract_archive_files**(archive_file_list: list, data_path: str

Extracts all given archive files (tars) into the dataset path.

* **archive_file_list**: (list) List of all archive files in the dataset path.
* **data_path**: (str) An absolute path to the dataset, e.g. ('D:\\KIA-datasets\\extracted').
* **delete_after_extract**: (bool) When true, archive files will be deleted after extraction.
* **dry_run**: (bool) When true, only output is enabled. No files will be modified.


---
### *def* **move_bit_tranche_1_seq_dirs**(data_path: str, dry_run: bool, delete_after_move: bool) -> None

Moves the BIT TS tranche 1 sequence directories to root path of dataset

* **data_path**: (str) An absolute path to the dataset, e.g. ('D:\\KIA-datasets\\extracted').
* **delete_after_move**: (bool) When true, sequence files will not moved but only copied.
* **dry_run**: (bool) When true, only output is enabled. No files will be modified.


---
### *def* **extract**() -> None

Entry point for kia_fix_extract.


