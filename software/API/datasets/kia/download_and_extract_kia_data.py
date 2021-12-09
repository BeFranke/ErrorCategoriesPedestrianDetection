from itertools import chain
from typing import List, Tuple, Dict, Any, Union
import json
import os
import subprocess
import argparse
import shutil
from collections import namedtuple


def download(remote_name: str, root_path: str, mc: str, env) -> None:
    remote_filelist = list_releases(remote_name, mc, env)
    missing_files = filter_downloads(remote_filelist)

    print("Missing releases: {}".format(len(missing_files)))
    for fname in missing_files:
        print(" - {}".format(fname))

    download_files(missing_files, remote_name, root_path, mc, env)


def list_releases(remote_name: str, mc: str, env) -> List[str]:

    process = subprocess.run([mc, 'ls', '{}/release'.format(remote_name)], stdout=subprocess.PIPE, env=env)
    process_output = process.stdout.decode("utf-8").split("\n")

    possible_filenames = [line.split(" ")[-1] for line in process_output]
    return sorted(set([filename for filename in possible_filenames if filename != ""]))


def filter_downloads(filenames: List[str]) -> List[str]:
    to_be_downloaded_files = sorted(frozenset(filenames))

    # Individual Filter - BitResultsSequences higher than x
    bit_sequences = [151, 152, 153, 154, 155, 156, 171, 172, 173]
    bit_files = [f for f in to_be_downloaded_files if f[:11] == 'bit_results' and int(f[21:25]) in bit_sequences]
    bit_files = [f for f in bit_files if f[-8:] not in ['x.tar.gz', 'h.tar.gz', 'r.tar.gz']]

    # # Mackevision - Sequences higher than 26 - No Tranche 2 from mv
    # macke_files = [f for f in to_be_downloaded_files if f[:10] == 'mv_results' and int(f[20:24]) > 26]
    #
    # # Concat
    # to_be_downloaded_files_filtered = bit_files + macke_files
    to_be_downloaded_files_filtered = bit_files

    return to_be_downloaded_files_filtered


def download_files(filenames: List[str],
                   remote_name: str,
                   root_path: str,
                   mc: str,
                   env) -> None:
    DownloadResult = namedtuple("DownloadResult", ["filename", "downloaded"])

    def _download(filename: str, root_path: str, env) -> DownloadResult:

        download_file_path = os.path.join(root_path, 'release', filename)

        if not os.path.isfile(download_file_path):
            print("Downloading: {}".format(filename))

            process = subprocess.run([mc, 'cp', '--continue',
                                      '{}/release/{}'.format(remote_name, filename), download_file_path],
                                     stdout=subprocess.PIPE, env=env)
            process_output = process.stdout.decode("utf-8")
            file_downloaded = os.path.isfile(download_file_path)

            return DownloadResult(filename=filename, downloaded=file_downloaded)

    # Dummy
    # filenames = ['bit_sequence_0179-21032ac691f24ce087ab3c4cc3a0b5fc.pdf']

    downloads, extractions = [], []
    for filename in filenames:
        # Download
        downloads.append(_download(filename, root_path, env))

        # Extraction
        extract_file(filename, root_path)

    successfully_downloaded = [download.filename for download in downloads if download.downloaded]
    failed_to_download = [download.filename for download in downloads if not download.downloaded]

    print("Downloads completed: {}".format(len(successfully_downloaded)))
    for filename in successfully_downloaded:
        print(" - {}\n".format(filename))

    print("Downloads failed: {}".format(len(failed_to_download)))
    for filename in failed_to_download:
        print(" - {}\n".format(filename))

    results_from_extractions = list(extractions)
    bad_results_from_extraction = [filename
                                   for filename, resultcode_ok
                                   in results_from_extractions if not resultcode_ok]

    if len(bad_results_from_extraction) > 0:
        print('Failed to move/extract:')
        for filename in bad_results_from_extraction:
            print('  - {}'.format(filename))

    print("Please inspect if extracting worked correctly.")


def extract_file(filename: str, root_path: str) -> None:
    def is_targz(s): return s.endswith(".tar.gz")
    def is_zip(s): return s.endswith(".zip")

    def extract_targz(filename: str, root_path: str) -> Tuple[str, bool]:
        return extracting(filename=filename,
                          subprocess_parameters=[
                              'tar',
                              '-C', os.path.join(root_path, 'extracted'),
                              '-xzvf', os.path.join(root_path, 'release', filename),
                          ])

    def extract_zip(filename: str, root_path: str) -> Tuple[str, bool]:
        return extracting(filename=filename,
                          subprocess_parameters=[
                              'unzip', '-d', os.path.join(root_path, 'extracted'),
                              os.path.join(root_path, 'release', filename),
                          ])

    def extracting(filename: str,
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
            os.path.join('ground-truth', 'depth_csv'),  # bit
            os.path.join('ground-truth', 'depth_png'),  # bit
            os.path.join('ground-truth', 'depth_exr'),  # mv
        ]

        for del_path in delete_paths:
            del_folder = os.path.join(extraction_folder, del_path)
            if os.path.exists(del_folder):
                shutil.rmtree('{}/{}'.format(extraction_folder, del_path), ignore_errors=True)

    if not os.path.exists(os.path.join(root_path, 'extracted')):
        os.mkdir(os.path.join(root_path, 'extracted'))

    extraction_file_path = os.path.join(root_path, 'extracted', filename)
    if is_targz(filename):
        extraction = extract_targz(filename, root_path)
        extracted_file_path = extraction_file_path.replace("tar.gz", "")
    elif is_zip(filename):
        extraction = extract_zip(filename, root_path)
        extracted_file_path = extraction_file_path.replace(".zip", "")
    else:
        # Move file to ./extracted
        shutil.move(src='b', dst='b')

    # Delete folders for convenience
    # delete_folders(extraction_file_path)

    # Move folder from extracted to home/opel/kia/input/datasets/kia


def move(root_path: str, move_path: str):

    filenames = os.listdir(os.path.join(root_path, 'extracted'))

    for filename in filenames:
        shutil.move(src=os.path.join(root_path, 'extracted', filename),
                    dst=os.path.join(move_path))

        print('Moved {}'.format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract')
    parser.add_argument('--remote_name', type=str, default="dsp", required=False, help='How you named your remote.')
    parser.add_argument('--mc', type=str, default="./mc", required=False)
    parser.add_argument('--root_path', type=str, default='../../media/hdd/kia', required=False)
    parser.add_argument('--move_path', type=str, default='./kia/input/datasets/kia', required=False)
    args = parser.parse_args()

    os.chdir(os.getenv("HOME"))
    print(os.getcwd())

    # Proxy for environment
    env = dict(os.environ)
    env['https_proxy'] = 'http://AZ1N73:OrlandoBassFishing_2020@10.81.68.10:8080'

    print("+--------------------------------------+")
    print("| Download and Extract latest releases |")
    print("+--------------------------------------+")

    download(args.remote_name, args.root_path, args.mc, env)
    move(args.root_path, args.move_path)
