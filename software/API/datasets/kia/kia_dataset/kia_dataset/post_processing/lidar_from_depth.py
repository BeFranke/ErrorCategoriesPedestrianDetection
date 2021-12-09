import os
from tqdm import tqdm
import numpy as np
import open3d as o3d
import cv2
import argparse


class PinholeCameraModel:
    """
    Container for camera intrinsics
    """
    def __init__(self, width=0, height=0, fx=1.0, fy=1.0, cx=0.0, cy=0.0):
        """

        :param width: Image width
        :param height: Image height
        :param fx: Focal length x
        :param fy: Focal length y
        :param cx: Optical center x
        :param cy: Optical center y
        """
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def read(self, path: str):
        """
        Read pinhole camera parameters form file
        :param path: Path to parameter file
        """
        with open(path, "r") as f:
            lines = f.readlines()
        width, height = [int(x) for x in re.match(".* ([0-9]+)x([0-9]+)px", lines[1]).groups()]
        K = np.array([[float(x) for x in y.strip(" ").split(" ")] for y in lines[10:13]])

        self.height = height
        self.width = width
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = width / 2.0
        self.cy = height / 2.0


class LidarFromDepth:
    """
    Creates pseudo-LIDAR pointclouds from depth images.
    """
    def __init__(self, start_row: int, start_col: int, end_row: int, end_col: int,
                 stride_rows: int, stride_cols: int, max_depth: float, intrinsics: PinholeCameraModel):
        """
        Initialize LIDAR parameters
        :param stride_rows: int Stride in y
        :param stride_cols: int Stride in x
        :param intrinsics: PinholeCameraModel Camera intrinsics
        """
        self.rows = intrinsics.height
        self.cols = intrinsics.width
        self.max_depth = max_depth
        self.intrinsics = intrinsics

        # Print parameters
        print(f"Camera Int: fx {intrinsics.fx:.2f}, fy {intrinsics.fy:.2f}, cx {intrinsics.cx:.2f}, cy {intrinsics.cy:.2f}")
        fov_x = 2.0 * np.arctan((intrinsics.width / 2.0) / intrinsics.fx) / np.pi * 180.0
        fov_y = 2.0 * np.arctan((intrinsics.height / 2.0) / intrinsics.fy) / np.pi * 180.0
        # Approximate angle resolution
        res_x = fov_x / intrinsics.width
        res_y = fov_y / intrinsics.height
        print(f"Camera FOV: {fov_x:.2f}° / {fov_y:.2f}°")
        print(f"Camera Res: {res_x:.2f}° / {res_y:.2f}°")

        lidar_width = (intrinsics.width - start_col - (end_col + 1))
        lidar_height = (intrinsics.height - start_row - (end_row + 1))
        fov_x = 2.0 * np.arctan((lidar_width / 2.0) / intrinsics.fx) / np.pi * 180.0
        fov_y = 2.0 * np.arctan((lidar_height / 2.0) / intrinsics.fy) / np.pi * 180.0
        # Approximate angle resolution
        res_x = fov_x / lidar_width * stride_cols
        res_y = fov_y / lidar_height * stride_rows
        print(f"LIDAR FOV: {fov_x:.2f}° / {fov_y:.2f}°")
        print(f"LIDAR Res: {res_x:.2f}° / {res_y:.2f}°")

        # Get pixel indices
        img_grid = np.array(np.meshgrid(np.arange(intrinsics.width), np.arange(intrinsics.height))).transpose((1, 2, 0))
        self.pixels = img_grid[start_row:end_row:stride_rows, start_col:end_col:stride_cols]

        # Backproject rays
        self.rays = np.ones(self.pixels.shape[:2] + (3,))
        self.rays[:, :, 0] = (self.pixels[:, :, 0] - intrinsics.cx) / intrinsics.fx
        self.rays[:, :, 1] = (self.pixels[:, :, 1] - intrinsics.cy) / intrinsics.fy

    def convert(self, depth: np.array) -> np.array:
        """
        Converts depth image into LIDAR pointcloud
        :param depth:  <height, width> Depth image
        :return: <row, col, 3> Structured LIDAR pointcloud
        """

        # Subsample depth image
        points = np.multiply(self.rays, np.tile(depth[self.pixels[:, :, 1], self.pixels[:, :, 0], np.newaxis], (1, 1, 3))).squeeze()

        # Apply max range
        points[points[:, :, 2] > self.max_depth] = -1.0

        # Transform pointcloud (optional)
        # TODO: Define LIDAR to camera transformation
        #points = np.multiply(self.T, points[:, :, np.newaxis]).squeeze()

        return points


def depth2pcd(camera, depth_folder, output_folder):
    # Initialize converter
    converter = LidarFromDepth(start_row=64, start_col=0, end_row=-1, end_col=-1,
                               stride_rows=16, stride_cols=3, max_depth=100.0, intrinsics=camera)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Go over all depth files
    files = os.listdir(depth_folder)
    for i, file in enumerate(tqdm(files, desc='{:10s}'.format("Processing"), leave=True, ascii=True)):
        output_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".pcd")
        if os.path.exists(output_path):
            continue

        # Read depth
        if file.endswith(".csv"):
            depth = np.genfromtxt(os.path.join(depth_folder, file), delimiter=',')[:, :-1]
        else:
            depth = cv2.imread(os.path.join(depth_folder, file), cv2.IMREAD_UNCHANGED)[:, :, 0]

        # Convert
        points = converter.convert(depth)

        # Drop invalid
        points = points[points[:, :, 2] >= 0.0]

        # Save pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(output_path, pcd)


def gen_lidar_from_depth(data_path):
    # Define intrinsics
    camera_bit = PinholeCameraModel(1920, 1080, 1677.0, 943.313, 960.0, 540.0)

    folders = os.listdir(data_path)
    for folder in folders:
        if folder.startswith("bit_results_") or folder.startswith("mv_results_"):
            print(f"Processing {folder}")
            # Check if exr files exist
            depth_dir = os.path.join(data_path, folder, "ground-truth", "depth_exr")
            # Else use csv files (BIT Tranche 1-2)
            if not os.path.exists(depth_dir):
                depth_dir = os.path.join(data_path, folder, "ground-truth", "depth_csv")
            lidar_dir = os.path.join(data_path, folder, "sensor", "lidar", "pcd")
            if not os.path.exists(lidar_dir) or len(os.listdir(lidar_dir)) == 0:
                depth2pcd(camera_bit, depth_dir, lidar_dir)


def main():
    parser = argparse.ArgumentParser(description='Add a fake lidar to the extracted data.')
    parser.add_argument('--data_path', type=str, required=True, help='The data path where the data was extracted.')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print("The path does not exist: {}".format(args.data_path))

    # Generate pseudo LIDAR from depthmaps
    gen_lidar_from_depth(args.data_path)


if __name__ == "__main__":
    main()
