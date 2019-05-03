import numpy as np
from mathutils import Quaternion, Vector
import json

def to_quat(x):
    return x if type(x) is Quaternion else x.to_quaternion()

def cartesian(*arrays):
    """Returns the cartesian product of multiple 1D arrays.

    For example, cartesian([0], [1, 2], [3, 4, 5]) returns:

    array([[0, 1, 3],
           [0, 1, 4],
           [0, 1, 5],
           [0, 2, 3],
           [0, 2, 4],
           [0, 2, 5]])

    Works with arbitrary objects.
    """
    # must do it this way to prevent numpy from turning any iterable into an np.array
    cleaned = np.empty(len(arrays), dtype=object)
    for i, array in enumerate(arrays):
        subcleaned = np.empty(len(array), dtype=object)
        subcleaned[:] = array
        cleaned[i] = subcleaned
    return np.stack(np.meshgrid(*cleaned), -1).reshape(-1, len(arrays))

def random_rotations(n):
    """
    Generates n rotations sampled uniformly from the group of all 3D rotations, SO(3).

    Args:
        n (int): number of rotations to generate

    Returns:
        List of mathutils.Quaternion objects.
    """
    wxyz = [np.random.normal(size=n) for _ in range(4)]
    return [Quaternion(t).normalized() for t in zip(*wxyz)]

def uniform_sphere(n, random=None):
    """
    Generates n points on the surface of a sphere that are "evenly spaced" using the golden spiral method. Based on
    https://stackoverflow.com/a/44164075.

    Args:
        n (int): number of points to generate over the surface of the sphere
        random (int): if None, return all generated points. Otherwise, randomly sample this many points from the
            generated ones (default: None)

    Returns:
        A tuple of the form (theta, phi), where theta and phi are each numpy arrays of length n. theta is the azimuthal
        angle, and phi is the polar angle.
    """
    indices = np.arange(0, n, dtype=np.float) + 0.5  # exclues start and endpoints while evenly spacing in between
    phi = np.arccos(2 * indices / n - 1)  # uniformly spaced along longitude lines
    theta = np.pi * (1 + 5**0.5) * indices % (2 * np.pi)  # golden spiral down sphere
    if random is None:
        return theta, phi
    else:
        return np.random.choice(theta, random), np.random.choice(theta, random)

def jsonify(obj):
    """Serializes an object's attributes into a JSON string with support for mathutils rotation objects.

    All rotation objects are converted to wxyz quaternion form.
    """
    d = vars(obj)
    for key, value in d.items():
        # handle rotations
        try:
            value = to_quat(value)
            d[key] = list(value)
        except AttributeError:
            if type(value) is Vector:
                d[key] = list(value)

    return json.dumps(d, indent=4)
