[Back to Overview](../README.md)



# kia_dataset.extractor

> Extract the tars downloaded by the downloader.

## Installation/Setup

Follow the instructions in the main readme.

## Running

Follow the instructions for the kia_extract command in the main readme.

Parameters:
* `--cache_path`: The place where to store the tars until they are extracted.
* `--data_path`: The path where the extracted data will be stored.
* `--delete_after_extract`: If True the data will be deleted immediately after successfull extraction.
This ensures a smaller memory footprint.


---
### *def* **extract_files**(data_path, cache_path, delete_after_extract) -> None

Extract all files in the folder.

* **cache_path**: (str) The place where to store the tars until they are extracted.
* **data_path**: (str) The path where the extracted data will be stored.
* **delete_after_extract**: (bool) If True the data will be deleted immediately after successfull extraction.
This ensures a smaller memory footprint.


---
### *def* **extract**()

Entry point for kia_extract.


