[Back to Overview](../README.md)



# kia_dataset.downloader

> Download the dataset from the DSP.

## Installation/Setup

Follow the instructions in the main readme.

## Running

Follow the instructions for the kia_fetch command in the main readme.

Parameters:
* `--cache_path`: The place where to store the tars until they are extracted.
* `--filename_download_journal`: A path to a file, where a journal of all already downloaded tars is kept.
Files in this journal will not be downloaded again. If you want to re-downlaod data for some reason remove the respective entry from that file.
* `--remote_name`: The name that you gave your remote in the setup of the minio client. If you follow the main readme setup process this will be kia.


---
### *def* **list_releases**(remote_name: str, keep_exr: bool, keep_depth: bool) -> List[str]

Get a list of all files on the DSP.

* **remote_name**: (str) The name that was given to the remote in the minio client setup.
* **keep_exr**: (bool) If the exr tars should be kept in the list.
* **keep_depth**: (bool) If the depth tars should be kept in the list.


---
### *def* **all_elements_are_strings**(listing): return all

*(no documentation found)*

---
### *def* **valid_json_form**(content): return type

*(no documentation found)*

---
### *def* **download**(filename: str) -> DownloadResult

*(no documentation found)*

---
### *def* **augment**(filename): return "{}".format(filename

*(no documentation found)*

---
### *def* **existing**(filename): return os.path.isfile(augment(filename)

*(no documentation found)*

---
### *def* **download_files**(filenames: List[str], remote_name: str, filename_download_journal: str) -> None

Downlaod the files in the list from the remote.

* **filenames**: (List[str]) A list of files that should be downloaded.
* **remote_name**: (str) The name that was given to the remote in the minio client setup.
* **filename_download_journal**: (str) The path to the file where the journal of downloaded files is kept.


---
### *def* **download_missing_files**(remote_name: str, filename_download_journal: str, download_exr: bool, download_depth: bool) -> None

Downlaod only missing files from the remote.

* **remote_name**: (str) The name that was given to the remote in the minio client setup.
* **filename_download_journal**: (str) The path to the file where the journal of downloaded files is kept.
* **download_exr**: (bool) If the exr tars should be downloaded as well.
* **download_depth**: (bool) If the depth tars should be downloaded as well.


---
### *def* **fetch**()

The entry point for kia_fetch.


