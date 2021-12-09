from itertools import chain
from typing import List, Tuple, Dict, Any, Union
import json
import os
import subprocess
import argparse
import shutil
from collections import namedtuple


def extract_files(download_dir: str) -> None:
    def is_targz(s): return s.endswith(".tar.gz")
    def is_zip(s): return s.endswith(".zip")

    def extract_targz(filename: str, download_dir: str) -> Tuple[str, bool]:
        return extracting(filename=filename,
                          download_dir=download_dir,
                          compression_suffix=".tar.gz",
                          subprocess_parameters=[
                              'tar',
                              '-C', 'extracted/',
                              '-xzvf', 'release/{}'.format(filename),
                          ])

    def extract_zip(filename: str, download_dir: str) -> Tuple[str, bool]:
        return extracting(filename=filename,
                          download_dir=download_dir,
                          compression_suffix=".zip",
                          subprocess_parameters=[
                              'unzip', '-d', 'extracted/', 'release/{}'.format(filename)])

    def extracting(filename: str,
                   download_dir: str,
                   compression_suffix: str,
                   subprocess_parameters: List[str]) -> Tuple[str, bool]:

        # process_dir = subprocess.run(['dir'], stdout=subprocess.PIPE)
        print("Extracting: {}".format(filename))
        result = subprocess.run(subprocess_parameters, stdout=subprocess.PIPE)
        return (filename, result.returncode == 0)

    def delete_folders(extraction_folder: str) -> None:
        delete_paths = [
            os.path.join('sensor', 'camera', 'left', 'exr'),
            os.path.join('sensor', 'lidar'),  # bit
            os.path.join('sensor', 'radar'),  # bit
            os.path.join('ground-truth', 'depth_exr'),  # vorher 2d bounding box fix

            # class_id
            # 2d-bounding-box-fixed_json
        ]

        for del_path in delete_paths:
            del_folder = os.path.join(extraction_folder, del_path)
            if os.path.exists(del_folder):
                shutil.rmtree('{}/{}'.format(extraction_folder, del_path), ignore_errors=True)

    # Already downloaded Sequences
    filenames = os.listdir(os.path.join(download_dir, 'release'))
    [print(filename) for filename in filenames]
    print('----------------------------------------------------')

    # Make extraction directory
    os.chdir(download_dir)
    # print(os.getcwd())
    if not os.path.exists('extracted'): os.mkdir('extracted')

    # Extract file
    extractions = []
    for filename in filenames:

        # Check for existing extractions
        if is_targz(filename):
            extraction_folder = filename.replace(".tar.gz", "")
        elif is_zip(filename):
            extraction_folder = filename.replace(".zip", "")
        else:
            AttributeError('Unknown compression type for {}'.format(filename))
        if os.path.exists(os.path.join(download_dir, 'extracted', extraction_folder)):
            print('{} is already extracted - Continue!'.format(filename.split('/')[-1]))
            continue

        # Extraction
        if is_targz(filename):
            extractions.append(extract_targz(filename, download_dir))
        elif is_zip(filename):
            extractions.append(extract_zip(filename, download_dir))
        else:
            AttributeError('Unknown compression type - Cannot extract!')

        # Delete folders for convenience
        delete_folders(extraction_folder)

    results_from_extractions = list(extractions)
    bad_results_from_extraction = [filename
                                   for filename, resultcode_ok
                                   in results_from_extractions if not resultcode_ok]

    if len(bad_results_from_extraction) > 0:
        print('Failed to move/extract:')
        for filename in bad_results_from_extraction:
            print('  - {}'.format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract')
    parser.add_argument('--download_dir', type=str, default="W:/1_projects/5_kia/software/kia_dataset/storage/", required=False)
    args = parser.parse_args()

    print("+-------------------------+")
    print("| Extract latest releases |")
    print("+-------------------------+")
    extract_files(args.download_dir)
