"""doc
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
"""

from itertools import chain
from typing import List, Tuple
import os
import subprocess
import argparse


def extract_files(data_path, cache_path, delete_after_extract) -> None:
    """
    Extract all files in the folder.

    :param cache_path: (str) The place where to store the tars until they are extracted.
    :param data_path: (str) The path where the extracted data will be stored.
    :param delete_after_extract: (bool) If True the data will be deleted immediately after successfull extraction.
        This ensures a smaller memory footprint.
    """
    def _is_targz(s): return s.endswith(".tar.gz")
    def _is_zip(s): return s.endswith(".zip")

    def _is_partial_download(s): return s.endswith(
        ".part.minio") or ".part." in s

    def _extracting(filename: str,
                   compression_suffix: str,
                   subprocess_parameters: List[str]) -> Tuple[str, bool]:
        relative_path = '{}/{}'.format(data_path, filename)
        extraction_folder = relative_path.replace(compression_suffix, "")

        if not os.path.exists(extraction_folder):
            print("Extracting: {}".format(filename))
            result = subprocess.run(
                subprocess_parameters, stdout=subprocess.PIPE)
            if delete_after_extract and result.returncode == 0:
                print('Deleting: {}/{}'.format(cache_path, filename))
                os.remove('{}/{}'.format(cache_path, filename))
            return (filename, result.returncode == 0)
        else:
            print('Not extracting: {} (already exists)'.format(relative_path))

        return (filename, True)

    def _extract_targz(filename: str) -> Tuple[str, bool]:
        return _extracting(filename=filename,
                          compression_suffix=".tar.gz",
                          subprocess_parameters=[
                              'tar', '-xzvf', '{}/{}'.format(cache_path, filename),
                              '--directory', '{}/'.format(data_path)])

    def _extract_zip(filename: str) -> Tuple[str, bool]:
        return _extracting(filename=filename,
                          compression_suffix=".zip",
                          subprocess_parameters=[
                              'unzip', '-d', '{}/'.format(data_path), '{}/{}'.format(cache_path, filename)])

    def _move(filename: str) -> Tuple[str, bool]:
        result = subprocess.run(
            ['mv', '{}/{}'.format(cache_path, filename), '{}/{}'.format(data_path, filename)], stdout=subprocess.PIPE)
        print(result.stdout.decode("utf-8"))
        return (filename, result.returncode == 0)

    files = os.listdir(cache_path)

    tars = frozenset(filter(_is_targz, files))
    zips = frozenset(filter(_is_zip, files))
    downloaded_uncompressed = frozenset(
        filter(lambda s: not _is_partial_download(s), files)) - tars - zips

    extractions = chain(
        map(_move, sorted(downloaded_uncompressed)),
        map(_extract_targz, sorted(tars)),
        map(_extract_zip, sorted(zips)))
    results_from_extractions = list(extractions)
    bad_results_from_extraction = [filename
                                   for filename, resultcode_ok
                                   in results_from_extractions if not resultcode_ok]

    if len(bad_results_from_extraction) > 0:
        print('Failed to move/extract:')
        for filename in bad_results_from_extraction:
            print('  - {}'.format(filename))

    if not delete_after_extract:
        print("Please inspect if extracting worked correctly and remove the tars/zips from {} manually.".format(cache_path))


def extract():
    """
    Entry point for kia_extract.
    """
    parser = argparse.ArgumentParser(description='Download and extract')
    parser.add_argument('--cache_path', type=str, required=True,
                        help='The path where the tar.gz are cached.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='The data path where the data should be extracted to extracted.')
    parser.add_argument('--delete_after_extract', action='store_true',
                        help='Flag, if you want to delete the tar directly after extracting.')
    args = parser.parse_args()
    extract_files(args.data_path, args.cache_path, args.delete_after_extract)
