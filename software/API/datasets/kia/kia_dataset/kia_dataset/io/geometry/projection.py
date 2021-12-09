"""doc
# kia_dataset.io.geometry.projection

> An implementation for projecting stuff.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer

**WARNING: This code is from dfki-dtk and might be removed in the future.**
"""
from typing import Tuple, Union
import math
import numpy as np
from kia_dataset.io.geometry.transform import Transform


def _solve_quadratic_equation(a, b, c):
    solution_1 = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    solution_2 = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return solution_1, solution_2


class Projection(object):
    def __init__(self, P, transform: Transform = None):
        """
        Create a projection given a projection matrix.

        :param P: The projection matrix of shape (3, 4). If the shape does not match it is resized. (A list of shape (12,) is also ok.)
        """
        if not isinstance(P, np.ndarray):
            P = np.array(P)
        self.transform = transform
        self.P = P.reshape(3, 4)

    def apply_inverse_to_point(self, points_uv: Union[np.ndarray], distances: np.ndarray) -> np.ndarray:
        """
        Apply the inverse of the projection to a point in uv space given the distance from the camera.

        Warning: This function only works if the P was the original projection matrix and all transforms have been done via the after interface.

        :param points_uv: 2d points of shape (2, N).
        :param distances: The distances from the camera of shape (1, N).
        :return: The point in 3d space (3, N).
        """
        points = []
        for i in range(points_uv.shape[1]):
            u, v = points_uv[0, i], points_uv[1, i]
            projection_matrix = self.P
            m00 = float(projection_matrix[0][0])
            m11 = float(projection_matrix[1][1])
            m02 = float(projection_matrix[0][2])
            m12 = float(projection_matrix[1][2])
            m22 = float(projection_matrix[2][2])
            t_1 = float(projection_matrix[0][3])
            t_2 = float(projection_matrix[1][3])
            t_3 = float(projection_matrix[2][3])

            alpha_1 = (u * m22 - m02) / m00
            alpha_2 = (v * m22 - m12) / m11
            beta_1 = (u * t_3 - t_1) / m00
            beta_2 = (v * t_3 - t_2) / m11

            a = 1 + alpha_1 * alpha_1 + alpha_2 * alpha_2
            b = 2 * (alpha_1 * beta_1 + alpha_2 * beta_2)
            c = beta_1 * beta_1 + beta_2 * beta_2 - distances[0, i] * distances[0, i]

            solution_1, solution_2 = _solve_quadratic_equation(a, b, c)

            z = solution_1 if solution_1 > solution_2 else solution_2
            x = alpha_1 * z + beta_1
            y = alpha_2 * z + beta_2
            points.append([x, y, z])
        points = np.array(points).T
        if self.transform is not None:
            points = self.transform.inverse.apply_to_point(points, single_point=False)
        return points

    def after(self, transform: Transform) -> 'Projection':
        """
        Apply a projection after a transformation.

        :param transform: The transform to apply before the projection.
        :return: The projection.
        """
        if self.transform is not None:
            t = transform.then(self.transform)
        else:
            t = transform
        return Projection(P=self.P, transform=t)

    def __call__(self, points):
        return self.apply_to_point(points)

    def apply_to_point(self, points: np.ndarray, single_point=False) -> Tuple[np.ndarray, Union[np.ndarray, float]]:
        """
        Apply a projection on a list of points.

        :param points: The points which to project. (3, N)
        :param single_point: If the result should be of shape (2,) and (1,) in case only one point is passed.
        :return: A tuple of the projected points and distances (negative is behind cam). ((2, N), (1, N))
        """
        points = np.array(points).reshape((3, -1))
        if self.transform is not None:
            points = self.transform(points, single_point=False)
        depth = np.sqrt(points[0, :] * points[0, :] + points[1, :] * points[1, :] + points[2, :] * points[2, :])
        homogenous_point = np.insert(points, 3, 1, axis=0)
        points_uv = np.dot(self.P, homogenous_point)
        behind_cam = points_uv[-1] <= 0
        points_uv = points_uv / points_uv[-1]
        points_uv = np.delete(points_uv, 2, axis=0)
        depth[behind_cam] *= -1
        if not single_point:
            return points_uv, depth
        else:
            return np.array([points_uv[0][0], points_uv[1][0]]), depth[0]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Projection(P={})".format(self.P)

    def pointcloud_to_depth_image(self, scan, reference_image_shape, scale=1.0, max_depth=100):
        """
        Project a point cloud to a depth image.

        :param scan: The point cloud of shape (3, N).
        :param reference_image_shape: A reference image for which the projection was made, to use the shape of it.
        :param scale: The scale compared to reference image.
        :param max_depth: The maximum allowed depth for the depth image.
        :return:
        """
        depth_image = np.zeros((int(reference_image_shape[0] * scale),
                                int(reference_image_shape[1] * scale), 1), dtype=np.float32)
        uv, depth = self.apply_to_point(scan)
        points = uv.T[depth > 0]
        filtered_scan = scan.T[depth > 0]
        depth = depth[depth > 0]
        for coord, point, d in zip(points, filtered_scan, depth):
            u = int(coord[0] * scale)
            v = int(coord[1] * scale)
            if d > max_depth:
                continue
            if 0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]:
                if d < depth_image[v, u] or depth_image[v, u] == 0:
                    depth_image[v, u] = d
        return depth_image

    def depth_image_to_pointcloud(self, depth_image, scale=1.0):
        """
        Create a pointcloud from a depth image.

        :param depth_image: A depth image.
        :param scale: The scale that was used to create the depth image.
        :return: A pointcloud of shape (3, N)
        """
        xy = np.indices(depth_image.shape[:2]) / scale
        xy = np.transpose(xy, (1, 2, 0))[..., ::-1]
        mask = (depth_image != 0)[:, :, 0]
        xy = xy[mask].T
        depth = depth_image[mask].T
        if xy.shape[1] == 0:
            return np.zeros((3, 0), dtype=np.float32)
        xyz = self.apply_inverse_to_point(xy, depth)
        return xyz
