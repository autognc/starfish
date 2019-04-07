import numpy as np
from mathutils import Quaternion

def to_quat(x):
    return x if type(x) is Quaternion else x.to_quaternion()

def generate_sphere_points(n):
    """
    Generates n points on the surface of a sphere that are "evenly spaced" using the golden spiral method. Based on
    https://stackoverflow.com/a/44164075.

    Args:
        n: int, number of points to generate

    Returns:
        A tuple of the form (theta, phi), where theta and phi are each numpy arrays of length n. theta is the azimuthal
        angle, and phi is the polar angle.
    """
    indices = np.arange(0, n, dtype=np.float) + 0.5  # exclues start and endpoints while evenly spacing in between
    phi = np.arccos(2 * indices / n - 1)  # uniformly spaced along longitude lines
    theta = np.pi * (1 + 5**0.5) * indices % (2 * np.pi)  # golden spiral down sphere
    return theta, phi
