"""doc
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
"""

import os
import sys
import shutil
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple

# ********************************************
# Helper functions
# ********************************************
def find_archive_files(data_path: str) -> list:
    """
    Finds all remaining archive files in the dataset path.

    :param data_path: (str) An absolute path to the dataset, e.g. ('D:\\KIA-datasets\\extracted').
    :return: (list) A list of all archive files in dataset path.
    """
    archive_files = []
    for file in os.listdir(data_path):
        if file.endswith(".tar"):
            archive_files.append(file)
    return archive_files

def extract_archive_files(archive_file_list: list, data_path: str,
                          delete_after_extract: bool, dry_run: bool):
    """
    Extracts all given archive files (tars) into the dataset path.

    :param archive_file_list: (list) List of all archive files in the dataset path.
    :param data_path: (str) An absolute path to the dataset, e.g. ('D:\\KIA-datasets\\extracted').
    :param delete_after_extract: (bool) When true, archive files will be deleted after extraction.
    :param dry_run: (bool) When true, only output is enabled. No files will be modified.
    """
    for archive_file in archive_file_list:
        filename, result_okay = _extract_tar(archive_file, data_path, delete_after_extract, dry_run)
        if(not result_okay):
            print('Failed to move/extract: {}'.format(filename))

def _extracting(filename: str,
                data_path: str,
                delete_after_extract: bool,
                dry_run: bool,
                compression_suffix: str,
                subprocess_parameters: List[str]) -> Tuple[str, bool]:
    
    relative_path = '{}/{}'.format(data_path, filename)
    extraction_folder = relative_path.replace(compression_suffix, "")
    """
    Extracts a given archive files into the dataset path.

    :param filename: (str) Filename of the archive file
    :param data_path: (str) An absolute path to the dataset, e.g. ('D:\\KIA-datasets\\extracted').
    :param delete_after_extract: (bool) When true, archive files will be deleted after extraction.
    :param dry_run: (bool) When true, only output is enabled. No files will be modified.
    :param compression_suffix: (str) The filetype, e.g. tar.
    :param subprocess_parameters: List[str] Commands to run as a sub process for extracting archive.

    :return: (Tuple[str, bool]) A tuple containing the filename of the archive and the result of the extraction process. True is successful.
    """
    if not os.path.exists(extraction_folder):
        print("Extracting: {}".format(filename))
        if not dry_run:
            result = subprocess.run(subprocess_parameters,
                                    stdout=subprocess.PIPE)
        else:
            result.returncode = 0
        if delete_after_extract and result.returncode == 0:
            print('Deleting: {}/{}'.format(data_path, filename))
            if not dry_run:
                os.remove('{}/{}'.format(data_path, filename))
        return (filename, result.returncode == 0)
    else:
        print('Not extracting: {} (already exists)'.format(relative_path))

    return (filename, True)

def _extract_tar(filename: str, data_path: str, delete_after_extract: bool, dry_run: bool) -> Tuple[str, bool]:
    """
    Extracts a given tar archive files into the dataset path.

    :param filename: (str) Filename of the archive file
    :param data_path: (str) An absolute path to the dataset, e.g. ('D:\\KIA-datasets\\extracted').
    :param delete_after_extract: (bool) When true, archive files will be deleted after extraction.
    :param dry_run: (bool) When true, only output is enabled. No files will be modified.

    :return: (Tuple[str, bool]) A tuple containing the filename of the archive and the result of the extraction process. True is successful.
    """
    return _extracting(filename=filename,
                       data_path=data_path,
                       delete_after_extract=delete_after_extract,
                       dry_run=dry_run,
                       compression_suffix=".tar",
                       subprocess_parameters=[
                           'tar', '-xvf', '{}/{}'.format(data_path, filename)])


def _parse_args():
    """
    Parses the given program arguments.

    Apart from parsing the program arguments, the function configures the
    logger used for outputs. Also it is checked if the arguments are correct.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Download and extract')
    parser.add_argument('-d', '--data_path', type=str, required=True, help='The data path where the data was extracted.')
    parser.add_argument('-n', '--no_delete', action='store_true', help='Do not delete archive files after extraction.')
    parser.add_argument('--dry_run', action='store_true', help='Dry-run mode will not change any files.')
    parser.add_argument('--logfile', type=str, help='When a log file is set, then file logging is enabled.')
    args = parser.parse_args()

    # Test if log file can be created
    try:
        Path(args.logfile).touch()
    except OSError as e:
        logging.error('Cannot create log file. Error: %s' % e)
        sys.exit()

    # file logging configuration
    log_format = ("[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s")
    if args.logfile:
        logging.basicConfig(filename=args.logfile, level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(level=logging.DEBUG, format=log_format)

    # check if data path exists
    if not os.path.exists(args.data_path):
        logging.error("The path does not exist: {}".format(args.data_path))
        logging.info("Please specificy an existing data path")
        sys.exit()

    return args

def move_bit_tranche_1_seq_dirs(data_path: str, dry_run: bool, delete_after_move: bool) -> None:
    """
    Moves the BIT TS tranche 1 sequence directories to root path of dataset

    :param data_path: (str) An absolute path to the dataset, e.g. ('D:\\KIA-datasets\\extracted').
    :param delete_after_move: (bool) When true, sequence files will not moved but only copied.
    :param dry_run: (bool) When true, only output is enabled. No files will be modified.
    """
    relative_bit_tranch_1_dir = os.path.join(data_path, "98-cluster-share/results_internal")
    if os.path.exists(relative_bit_tranch_1_dir):
        bit_tranch_1_seq_dirs = os.listdir(relative_bit_tranch_1_dir)
        for sequence_dir in bit_tranch_1_seq_dirs:
            logging.info("Move {0}/{1} to {2}/{1}".format(relative_bit_tranch_1_dir,
                                                sequence_dir, data_path))
            if not dry_run:
                try:
                    shutil.copytree("{}/{}".format(relative_bit_tranch_1_dir, sequence_dir),
                                    "{}/{}".format(data_path, sequence_dir))
                except shutil.Error as e:
                    logging.error('Directory not copied. Error: %s' % e)
                except OSError as e:
                    logging.error('Directory not copied. Error: %s' % e)
        if delete_after_move:
            realtive_bit_tranch_1_root_dir = os.path.join(data_path, "98-cluster-share")
            logging.info("Remove {}".format(realtive_bit_tranch_1_root_dir))
            if not dry_run:
                shutil.rmtree(realtive_bit_tranch_1_root_dir)
    else:
        logging.info("The path does not exists: {}".format(relative_bit_tranch_1_dir))

# ********************************************
# Entry points to script
# ********************************************
def extract() -> None:
    """
    Entry point for kia_fix_extract.
    """
    
    logging.info("### Fixing data extraction")
    args = _parse_args()
    os.chdir(args.data_path)

    if args.dry_run:
        logging.info("Dry-run enabled. No files will be modified")
    logging.info("Find archive files")
    archive_files = find_archive_files(args.data_path)
    if archive_files:
        logging.info("Extract archive files")
        extract_archive_files(archive_files, args.data_path, not args.no_delete, args.dry_run)
    else:
        logging.info("No archive files found")
    logging.info("Move bit tranche 1 sequence directories")
    move_bit_tranche_1_seq_dirs(args.data_path, args.dry_run, not args.no_delete)
    logging.info("### Fix done")

if __name__ == "__main__":
    extract()