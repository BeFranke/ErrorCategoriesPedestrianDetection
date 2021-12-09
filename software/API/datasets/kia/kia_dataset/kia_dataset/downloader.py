"""doc
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
"""
from typing import List, Tuple
import json
import os
import subprocess
import argparse
from collections import namedtuple

MINIO_CLIENT_COMMAND = "mc"
# FILENAME_EXTRACTION_JOURNAL = "extracted.json"


def list_releases(remote_name: str, keep_exr: bool, keep_depth: bool) -> List[str]:
    """
    Get a list of all files on the DSP.

    :param remote_name: (str) The name that was given to the remote in the minio client setup.
    :param keep_exr: (bool) If the exr tars should be kept in the list.
    :param keep_depth: (bool) If the depth tars should be kept in the list.
    """
    process = subprocess.run(
        [MINIO_CLIENT_COMMAND, 'ls', remote_name + '/release'], stdout=subprocess.PIPE)
    process_output = process.stdout.decode("utf-8").split("\n")

    possible_filenames = [line.split(" ")[-1] for line in process_output]
    if not keep_exr:
        print("Skipping exr")
        possible_filenames = [x for x in possible_filenames if not x.endswith("exr.tar.gz")]
    if not keep_depth:
        print("Skipping depth")
        possible_filenames = [x for x in possible_filenames if not x.endswith("depth.tar.gz")]
    return sorted(set([filename for filename in possible_filenames if filename != ""]))


def _read_str_list_json(filename: str) -> List[str]:
    def all_elements_are_strings(listing): return all(
        [type(element) == str for element in listing])
    def valid_json_form(content): return type(
        content) == list and all_elements_are_strings(content)

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            content = json.loads(f.read())
            if not valid_json_form(content):
                raise Exception(
                    'JSON file {} is malformed -> allowed is a list of strings only!'.format(content))
            return content
    else:
        raise Exception('could not find JSON file {}!'.format(filename))


def _read_journal(filename_journal: str) -> List[str]:
    return _read_str_list_json(filename_journal) if os.path.isfile(filename_journal) else []


def _write_journal(filename_journal: str, filenames: List[str]) -> None:
    print('>>> writing extracted journal - DO NOT ABORT PROGRAM! <<<')
    with open(filename_journal, 'w') as f:
        f.write(json.dumps(filenames, indent=4))
    print('>>> extraction journal written <<<')


def _extend_journal(filename_journal: str, filenames: List[str]) -> None:
    journal = _read_journal(filename_journal)
    journal.extend(filenames)
    _write_journal(filename_journal, sorted(set(journal)))


def _downloaded_filenames_filtered_out_from(filenames: List[str], filename_download_journal: str) -> List[str]:
    downloaded_files = _read_journal(filename_download_journal)
    to_be_downloaded_files = sorted(
        frozenset(filenames) - frozenset(downloaded_files))
    return to_be_downloaded_files


def _download_safe_filenames_with_result(filenames: List[str], remote_name: str, filename_download_journal: str) -> Tuple[List[str], List[str]]:
    DownloadResult = namedtuple("DownloadResult", ["filename", "downloaded"])
    def download(filename: str) -> DownloadResult:
        def augment(filename): return "{}".format(filename)
        def existing(filename): return os.path.isfile(augment(filename))

        print("Downloading: {}".format(filename))
        process = subprocess.run([MINIO_CLIENT_COMMAND, 'cp', '--continue', '{}/release/{}'.format(
            remote_name, filename), augment(filename)], stdout=subprocess.PIPE)
        process_output = process.stdout.decode("utf-8")
        print(process_output)

        file_downloaded = existing(filename)
        if file_downloaded:
            _extend_journal(filename_download_journal, [filename])

        return DownloadResult(filename=filename, downloaded=file_downloaded)

    downloads = [download(filename) for filename in filenames]
    successfully_downloaded = [download.filename for download in downloads if download.downloaded]
    failed_to_download = [download.filename for download in downloads if not download.downloaded]

    return successfully_downloaded, failed_to_download


def download_files(filenames: List[str], remote_name: str, filename_download_journal: str) -> None:
    """
    Downlaod the files in the list from the remote.

    :param filenames: (List[str]) A list of files that should be downloaded.
    :param remote_name: (str) The name that was given to the remote in the minio client setup.
    :param filename_download_journal: (str) The path to the file where the journal of downloaded files is kept.
    """
    proper_filenames = [filename for filename in filenames if filename != ""]
    to_be_downloaded_files = _downloaded_filenames_filtered_out_from(
        proper_filenames, filename_download_journal)

    successfully_downloaded, failed_to_download = _download_safe_filenames_with_result(
        to_be_downloaded_files, remote_name, filename_download_journal)

    print("Downloads completed: {}".format(len(successfully_downloaded)))
    for filename in successfully_downloaded:
        print(" - {}\n".format(filename))

    print("Downloads failed: {}".format(len(failed_to_download)))
    for filename in failed_to_download:
        print(" - {}\n".format(filename))


def download_missing_files(remote_name: str, filename_download_journal: str, download_exr: bool, download_depth: bool) -> None:
    """
    Downlaod only missing files from the remote.

    :param remote_name: (str) The name that was given to the remote in the minio client setup.
    :param filename_download_journal: (str) The path to the file where the journal of downloaded files is kept.
    :param download_exr: (bool) If the exr tars should be downloaded as well.
    :param download_depth: (bool) If the depth tars should be downloaded as well.
    """
    remote_filelist = list_releases(remote_name, download_exr, download_depth)
    missing_files = _downloaded_filenames_filtered_out_from(remote_filelist, filename_download_journal)

    print("Missing releases: {}".format(len(missing_files)))
    for fname in missing_files:
        print(" - {}".format(fname))

    download_files(missing_files, remote_name, filename_download_journal)


def fetch():
    """
    The entry point for kia_fetch.
    """
    parser = argparse.ArgumentParser(description='Download and extract')
    parser.add_argument('--cache_path', type=str, required=True, help='The path where the tar.gz are cached.')
    parser.add_argument('--remote_name', type=str, default="kia", required=False, help='How you named your remote.')
    parser.add_argument('--filename_download_journal', type=str, default="downloaded.json", required=False, help='The filename for the download journal.')
    parser.add_argument('--download_exr', action='store_true', help='If the exr should be downloaded as well.')
    parser.add_argument('--download_depth', action='store_true', help='If the depth data should be downloaded as well.')
    args = parser.parse_args()
    os.chdir(args.cache_path)
    download_missing_files(args.remote_name, args.filename_download_journal, args.download_exr, args.download_depth)
