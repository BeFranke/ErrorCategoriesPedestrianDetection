[Back to Overview](../../../README.md)



# kia_dataset.io.geometry.projection

> An implementation for projecting stuff.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer

**WARNING: This code is from dfki-dtk and might be removed in the future.**


---
---
## *class* **Projection**(object)

Create a projection given a projection matrix.

* **P**: The projection matrix of shape (3, 4). If the shape does not match it is resized. (A list of shape (12,) is also ok.)


---
### *def* **apply_inverse_to_point**(*self*, points_uv: Union[np.ndarray], distances: np.ndarray) -> np.ndarray

Apply the inverse of the projection to a point in uv space given the distance from the camera.

Warning: This function only works if the P was the original projection matrix and all transforms have been done via the after interface.

* **points_uv**: 2d points of shape (2, N).
* **distances**: The distances from the camera of shape (1, N).
* **returns**: The point in 3d space (3, N).


---
### *def* **after**(*self*, transform: Transform) -> 'Projection'

Apply a projection after a transformation.

* **transform**: The transform to apply before the projection.
* **returns**: The projection.


---
### *def* **apply_to_point**(*self*, points: np.ndarray, single_point=False) -> Tuple[np.ndarray, Union[np.ndarray, float]]

Apply a projection on a list of points.

* **points**: The points which to project. (3, N)
* **single_point**: If the result should be of shape (2,) and (1,) in case only one point is passed.
* **returns**: A tuple of the projected points and distances (negative is behind cam). ((2, N), (1, N))


---
### *def* **pointcloud_to_depth_image**(*self*, scan, reference_image_shape, scale=1.0, max_depth=100)

Project a point cloud to a depth image.

* **scan**: The point cloud of shape (3, N).
* **reference_image_shape**: A reference image for which the projection was made, to use the shape of it.
* **scale**: The scale compared to reference image.
* **max_depth**: The maximum allowed depth for the depth image.
* **returns**:


---
### *def* **depth_image_to_pointcloud**(*self*, depth_image, scale=1.0)

Create a pointcloud from a depth image.

* **depth_image**: A depth image.
* **scale**: The scale that was used to create the depth image.
* **returns**: A pointcloud of shape (3, N)


