"""doc
# kia_dataset.io.geometry.helpers

> Little geometry helper functions.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer

**WARNING: This code is from dfki-dtk and might be removed in the future.**
"""
import math


def magnitude_of_vec(x):
    """
    Compute the magnitude of a 3 dimensional vector (array with 3 entries).
    
    :param x: The list representing a vector.
    :return: The magnitude of the vector (euclidean length).
    """
    return math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])


def normalize_vec(x):
    """
    Normalize a 3 dimensional vector (array with 3 entries) to unit length.
    
    :param x: The list representing a vector that should be normalized.
    :return: The normalized vector x.
    """
    magnitude = magnitude_of_vec(x)
    return [x[0] / magnitude, x[1] / magnitude, x[2] / magnitude]
