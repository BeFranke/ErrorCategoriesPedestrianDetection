"""doc
# kia_dataset.io.geometry.transform

> Geometric transformations between coordinate systems.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer

**WARNING: This code is from dfki-dtk and might be removed in the future.**
"""
from typing import Union, Sequence
import numpy as np
from pyquaternion import Quaternion


class Transform(object):
    @staticmethod
    def identity():
        return Transform(t=[0.0, 0.0, 0.0], R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    def __init__(self, t=None, q=None, R=None, Rt=None):
        if t is not None:
            if not isinstance(t, np.ndarray):
                t = np.array(t)
            if q is not None:
                if not isinstance(q, Quaternion):
                    q = Quaternion(q)
                q = q.normalised
                R = q.rotation_matrix
            elif R is not None:
                if not isinstance(R, np.ndarray):
                    R = np.array(R)
            else:
                #warn("You did not specify a rotation, was this intentional? You should provide R or q parameter.")
                R = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            R = R.reshape(3, 3)
            t = t.reshape(3, 1)
            Rt = np.hstack([R, t])
            Rt = np.vstack([Rt, [0, 0, 0, 1]])
            self.Rt = Rt
        elif Rt is not None:
            self.Rt = Rt
        else:
            raise RuntimeError("You must specify either Rt or the translation t with a rotation R/q.")

    @property
    def inverse(self):
        R = self.R.T
        t = -self.R.T.dot(self.t).reshape(3, 1)
        Rt = np.hstack([R, t])
        Rt = np.vstack([Rt, [0, 0, 0, 1]])
        return Transform(Rt=Rt)

    @property
    def R(self):
        return self.Rt[:3, :3]

    @property
    def t(self):
        return self.Rt[:3, 3:].reshape(3)

    @property
    def q(self):
        return Quaternion(matrix=self.Rt)

    @property
    def axis(self):
        return self.q.axis

    @property
    def angle(self):
        return self.q.angle

    @property
    def radians(self):
        return self.q.radians

    @property
    def degrees(self):
        return self.q.degrees

    def then(self, transform):
        return Transform(Rt=np.matmul(transform.Rt, self.Rt))

    def then_rotate(self, R=None, q=None):
        transform = Transform(R=R, q=q, t=[0, 0, 0])
        return Transform(Rt=np.matmul(transform.Rt, self.Rt))

    def __call__(self, point, single_point=True):
        return self.apply_to_point(point, single_point=single_point)

    def apply_to_point(self, point, single_point=True):
        """
        Apply transform to a pointcloud of shape (3, N).

        :param point: Return a pointcloud of shape (3, N).
        :param single_point: Boolean if a single point should be processed. (Output then is only (3,).)
        :return: The transformed pointcloud of shape (3, N).
        """
        homogenous_point = np.insert(np.array(point).reshape((3, -1)), 3, 1, axis=0)
        result = np.dot(self.Rt, homogenous_point)
        if single_point:
            return np.array([result[0][0], result[1][0], result[2][0]])
        else:
            return result[0:3, :]

    def apply_to_direction(self, direction):
        homogenous_direction = np.insert(np.array(direction).reshape((3, -1)), 3, 0, axis=0)
        result = np.dot(self.Rt, homogenous_direction)
        return np.array([result[0][0], result[1][0], result[2][0]])

    def apply_to_orientation(self, orientation: Union[Quaternion, Sequence[float]]) -> Quaternion:
        if not isinstance(orientation, Quaternion):
            orientation = Quaternion(orientation)
        transform = Transform(t=[0,0,0], q = orientation)
        return transform.then(self).q

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Transform(Rt={})".format(self.Rt)

    @property
    def unsafe_q(self):
        return Quaternion(matrix=self.Rt, rtol=1E-5, atol=1E-5)
