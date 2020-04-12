import numpy as np
from mathutils import Quaternion, Vector

from .utils import to_quat

"""
This module is for alternative 3D rotation formalisms besides the Quaternion, Matrix, and Euler representations provided
by the mathutils library. They must implement the `to_quaternion` method, which returns a mathutils.Quaternion instance,
in order to be compatible with the rest of this library. A `from_other` classmethod may also be useful, in order to
convert from a mathutils representation to the alternative representation.
"""


class Spherical:
    """An alternative 3-value representation of a rotation based on spherical coordinates.

    Imagine a unit sphere centered about an object. Two spherical coordinates (an azimuthal angle, henceforth theta, and
    a polar angle, henceforth phi) define a point on the surface of the sphere, and a corresponding unit vector from the
    center of the sphere (the object) to the point on the surface of the sphere.

    First, the +Z axis of the object is rotated to this unit vector, while the XY plane of the object is aligned such
    that the +X axis points in the +phi direction and the +Y axis points in the +theta direction. It may be helpful to
    visualize this rotation as such: imagine that the +Z axis of the object is a metal rod attached rigidly to the
    object, extending out through the surface of the sphere. Now grab the rod and use it to rotate the object such that
    the rod is passing through the point on the sphere defined by theta and phi. Finally, twist the rod such that the
    original "right" direction of the object (its +X axis) is pointing towards the south pole of the sphere, along the
    longitude line defined by theta. Correspondingly, this should mean that the original "up" direction of the object
    (its +Y axis) is pointing eastward along the latitude line defined by phi.

    Next, perform a right-hand rotation of the object about the same unit vector by a third angle (henceforth called the
    roll angle). In the previous analogy, this is equivalent to then twisting the metal rod counter-clockwise by the
    roll angle. This configuration is the final result of the rotation.

    Note: the particular alignment of the XY plane (+X is +phi and +Y is +theta) was chosen so that "zero rotation"
    (aka the identity quaternion, or (0, 0, 0) Euler angles) corresponds to (theta, phi, roll) = (0, 0, 0).

    Also note that this representation only uses 3 values, and thus it has singularities at the poles where theta and
    the roll angle are redundant (only their sum matters).

    Attributes:
        theta: The azimuthal angle, in radians
        phi: The polar angle, in radians (0 at the north pole, pi at the south pole)
        roll: The roll angle, in radians
    """

    def __init__(self, theta, phi, roll):
        """
        Initializes a spherical rotation object.

        Args:
            theta: The azimuthal angle, in radians
            phi: The polar angle, in radians (0 at the north pole, pi at the south pole)
            roll: The roll angle, in radians
        """
        self.theta = theta % (2 * np.pi)
        self.phi = phi % (2 * np.pi)
        self.roll = roll % (2 * np.pi)

    @classmethod
    def from_other(cls, obj):
        """
        Constructs a Spherical object from a Quaternion, Euler, or Matrix rotation object from the mathutils library.
        """
        if type(obj) is cls:
            return obj
        obj = to_quat(obj)

        # first, rotate the +Z unit vector by the object to replicate the effect of rot_quat without roll_quat
        z_axis = Vector((0, 0, 1))
        z_axis.rotate(obj)
        # calculate the inverse of rot_quat, which is the rotation that moves the new position of the unit vector to
        # its original position on the +Z axis
        inv_rot_quat = Quaternion(z_axis.cross(Vector((0, 0, 1))), z_axis.angle(Vector((0, 0, 1))))
        # extract roll_quat by left-multiplying by the inverse of rot_quat
        roll_quat = inv_rot_quat @ obj

        # calculate theta and phi from the new position of the unit vector, as well as roll directly from roll_quat
        theta = np.arctan2(z_axis.y, z_axis.x)
        phi = np.arccos(np.clip(z_axis.z, -1, 1))
        roll = roll_quat.to_euler().z - theta
        return cls(theta, phi, roll)

    def to_quaternion(self):
        """Returns a mathutils.Quaternion representation of the rotation."""
        # first, rotate about the +Z axis by the roll angle plus theta to align the +X axis with +phi and the +Y axis
        # with +theta
        roll_quat = Quaternion((0, 0, 1), self.roll + self.theta)
        # then, rotate the +Z axis to the unit vector represented by theta and phi by rotating by phi about a vector
        # tangent to the sphere pointing in the +theta direction
        theta_tangent = (-np.sin(self.theta), np.cos(self.theta), 0)
        rot_quat = Quaternion(theta_tangent, self.phi)
        # compose the two rotations and return
        return rot_quat @ roll_quat

    def __eq__(self, other):
        return self.theta == other.theta and self.phi == other.phi and self.roll == other.roll

    def __repr__(self):
        return f"<Spherical (theta={self.theta}, phi={self.phi}, roll={self.roll})>"
