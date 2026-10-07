"""Test helper: build a src.affine_fit.RigidTransform from rotation + translation."""
import numpy as np

from src.affine_fit import RigidTransform


def rigid_transform(rotation: float, translation) -> RigidTransform:
    """Rigid transform from a rotation angle (radians) and a (tx, ty) translation
    (the matrix skimage's ``EuclideanTransform(rotation=, translation=)`` builds)."""
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    params = np.eye(3)
    params[:2, :2] = [[cos_r, -sin_r], [sin_r, cos_r]]
    params[0, 2], params[1, 2] = translation
    return RigidTransform(params)
