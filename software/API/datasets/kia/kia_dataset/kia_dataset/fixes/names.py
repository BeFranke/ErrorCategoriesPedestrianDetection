"""doc
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
"""

import os
import shutil
import argparse


# ********************************************
# Helper functions
# ********************************************
def _recursive_rename_prefix(search, name_fixer, data_path, folder_contains=""):
    # Find all files that match a search prefix and are within a folder where the name contains folder_contains.
    # On all files that match apply a name_fixer=lambda fname: return fname
    for f in os.listdir(data_path):
        if f.startswith(search) and folder_contains in data_path:
            old_f = f
            f = name_fixer(f)
            shutil.move(os.path.join(data_path, old_f), os.path.join(data_path, f))

        fullpath = os.path.join(data_path, f)
        if os.path.isdir(fullpath):
            _recursive_rename_prefix(search, name_fixer, fullpath, folder_contains)


# ********************************************
# All Tranches
# ********************************************
def clean_emtpy_folders(data_path):
    """
    Remove all empty folders from the data.
    
    :param data_path: (str) The path where the extracted data is.
    """
    # Clean empty folders from subfolders (as this might make this folder empty)
    for f in os.listdir(data_path):
        fullpath = os.path.join(data_path, f)
        if os.path.isdir(fullpath):
            clean_emtpy_folders(fullpath)

    # Delete all folders that are empty
    if len(os.listdir(data_path)) == 0:
        os.rmdir(data_path)


# ********************************************
# Tranche 1
# ********************************************
def fix_sequence_folder_prefix_BIT_Tranche1(data_path):
    """
    Add the bit prefix to the data of tranche 1 from bit.
    
    :param data_path: (str) The path where the extracted data is.
    """
    # Rename from results_ to bit_results_
    folders = os.listdir(data_path)
    for folder in folders:
        if folder.startswith("results_"):
            old_folder = os.path.join(data_path, folder)
            new_folder = os.path.join(data_path, folder.replace("results_", "bit_results_"))
            print("Moving {} -> {}".format(old_folder, new_folder))
            shutil.move(old_folder, new_folder)


def fix_filename_scheme_MV_Tranche1(data_path):
    """
    Rename the ground truth in MV tranche 1 from "ground_truth" to "ground-truth"
    
    :param data_path: (str) The path where the extracted data is.
    """
    # Fix ground-truth name scheme
    folders = os.listdir(data_path)
    for folder in folders:
        old_folder = os.path.join(data_path, folder, "ground_truth")
        new_folder = os.path.join(data_path, folder, "ground-truth")
        if os.path.exists(old_folder):
            print("Moving {} -> {}".format(old_folder, new_folder))
            shutil.move(old_folder, new_folder)

    # Fix camera name scheme (arb-, car- prefixing)
    _recursive_rename_prefix("CameraE", lambda fname: fname.replace(
        "CameraE", "car-camera").replace("_sequence_", "_").replace(".", "-", 1), data_path)
    _recursive_rename_prefix("Camera", lambda fname: fname.replace(
        "Camera", "arb-camera").replace("_sequence_", "_").replace(".", "-", 1), data_path)


# ********************************************
# Tranche 2
# ********************************************
def fix_filename_scheme_BIT_Tranche2(data_path):
    """
    In tranche 2 of BIT fix the prefix for global annotations to "world-" from "sequence_".
    
    :param data_path: (str) The path where the extracted data is.
    """
    # sequence_ prefix changed to world- prefix
    _recursive_rename_prefix("sequence_", lambda fname: fname.replace(
        "sequence_", "world-"), data_path, folder_contains="ground-truth")


# ********************************************
# Entry points to script
# ********************************************
def _parse_args():
    parser = argparse.ArgumentParser(description='Download and extract')
    parser.add_argument('--data_path', type=str, required=True, help='The data path where the data was extracted.')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print("The path does not exist: {}".format(args.data_path))

    return args

def tranche_1():
    """
    Entry point for kia_fix_names_tranche_1.
    """
    args = _parse_args()
    print("Removing empty folders.")
    clean_emtpy_folders(args.data_path)
    print("Fix folder prefix for tranche 1. (BIT)")
    # rename "results_" to "bit_results_"
    fix_sequence_folder_prefix_BIT_Tranche1(args.data_path)
    print("Fix filename scheme for tranche 1. (MV)")
    # rename "ground_truth" to "ground-truth" and "CameraE" to "car-camera" and "Camera" to "arb-camera" and some minor stuff
    fix_filename_scheme_MV_Tranche1(args.data_path)


def tranche_2():
    """
    Entry point for kia_fix_names_tranche_2.
    """
    args = _parse_args()
    print("Removing empty folders.")
    clean_emtpy_folders(args.data_path)
    print("Fix filename scheme for tranche 2. (BIT)")
    # rename "sequence_" to "world-"
    fix_filename_scheme_BIT_Tranche2(args.data_path)
