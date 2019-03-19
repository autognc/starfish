import bpy
from mathutils import Quaternion, Vector
import numpy as np

def to_quat(x):
    return x if type(x) is Quaternion else x.to_quaternion()

class Spherical:
    """
    An alternative 3-value representation of a rotation based on spherical coordinates.

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
    def from_other(this, obj):
        """
        Constructs a Spherical object from a Quaternion, Euler, or Matrix rotation object from the mathutils library.
        """
        if type(obj) is this:
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
        return this(theta, phi, roll)


    def __matmul__(self, other):
        """
        Composes this rotation with another rotation object (Spherical, Euler, Quaternion, or Matrix).
        """
        other = to_quat(other)
        return Spherical.from_other(self.to_quaternion() @ other)

    def to_quaternion(self):
        """
        Returns a mathutils.Quaternion representation of the rotation.
        """
        # first, rotate about the +Z axis by the roll angle plus theta to align the +X axis with +phi and the +Y axis
        # with +theta
        roll_quat = Quaternion((0, 0, 1), self.roll + self.theta)
        # then, rotate the +Z axis to the unit vector represented by theta and phi by rotating by phi about a vector
        # tangent to the sphere pointing in the +theta direction
        theta_tangent = (-np.sin(self.theta), np.cos(self.theta), 0)
        rot_quat = Quaternion(theta_tangent, self.phi)
        # compose the two rotations and return
        return rot_quat @ roll_quat

    def __repr__(self):
        return f"<Spherical (theta={self.theta}, phi={self.phi}, roll={self.roll})>"


class Picture:
    """
    Represents a single picture of an object with certain parameters.

    There are 6 parameters that independently define a picture:
        1. Object position: the absolute 3D position of the object in the global coordinate system (i.e. where it is in
           the scene).
        2. Camera distance: the distance of the camera from the object.
        3. Object pose: the pose of the object relative to the camera (i.e. how it will appear to be oriented in the
           picture).
        4. Lighting: the angle from which the sun's rays will hit the object in the picture (e.g. from above, from the
           right, from behind, etc.).
        5. Object offset: the 2D translational offset of the object from the center of the picture frame.
        6. Background/camera orientation: the orientation of the camera relative to the global coordinate system. This
           affects only what part of the scene appears in the background and at what angle it appears.
    """

    def __init__(self, position=(0, 0, 0), distance=100, pose=Quaternion(),
                 lighting=Quaternion(), offset=(0.5, 0.5), background=Quaternion()):
        """
        Initializes a picture with all of the parameters it needs.

        A type of "rotation" means any of Spherical, mathutils.Quaternion, mathutils.Euler, or mathutils.Matrix are
        acceptable.

        Args:
            position (seq, len 3): the (x, y, z) absolute position of the object in the scene (default: (0, 0, 0))
            distance (float or int): the distance of the camera from the object in blender units (default: 100)
            pose (rotation): the pose of the object relative to the camera (default: the identity quaternion (aka zero
                rotation), which corresponds to the camera looking directly in the object's -Z direction with the
                object's +X direction pointing up and +Y pointing to the right)
            lighting (rotation): the angle of the sun's lighting relative to the camera (default: the identity
                quaternion (aka zero rotation), which corresponds to the light coming from directly behind the camera)
            offset (seq of float, len 2): the (horizontal, vertical) translational offset of the object from the center
                of the picture frame. Expressed as a fraction of the distance from edge to edge: e.g., for horizontal
                offset, 0.0 is the left edge, 0.5 is the center, and 1.0 is the right edge. Same for vertical, but 0.0
                is the bottom edge and 1.0 is the top. (default: (0.5, 0.5))
            background (rotation): the orientation of the camera relative to the global coordinate system (default: the
                identity quaternion (aka zero rotation), which corresponds to the camera looking directly in the -Z
                direction with the +X direction pointing up and +Y pointing to the right)
        """
        self.position = Vector(position)
        self.distance = distance
        self.pose = to_quat(pose)
        self.lighting = to_quat(lighting)
        self.offset = tuple(offset)
        self.background = Spherical.from_other(background)

    def setup(self, obj, camera, sun):
        """
        Sets up a camera, object, and sun into the picture-taking position.

        Args:
            obj (BlendDataObject): the object that will be the subject of the picture
            camera (BlendDataObject): the camera to take the picture with
            sun (BlendDataObject): the sun lamp that is providing the lighting
        """
        # set object position
        obj.location = self.position

        # set camera position
        cartesian = self.distance * Vector((np.sin(self.background.phi) * np.cos(self.background.theta),
                                            np.sin(self.background.phi) * np.sin(self.background.theta),
                                            np.cos(self.background.phi)))
        camera.location = cartesian + self.position

        # set camera rotation
        camera.rotation_mode = "QUATERNION"
        camera.rotation_quaternion = self.background.to_quaternion()

        # modify camera rotation to produce correct object offset
        x_frac, y_frac = self.offset
        # convert proporitions to angle offsets using camera FOV
        # FIXME: the FOV values given by Blender, usually angle_y, are a little off sometimes. It's good enough for
        #        now, but maybe there's another way to compute the angles that doesn't have this bug
        x_angle = (x_frac - 0.5) * camera.data.angle_x
        y_angle = (y_frac - 0.5) * camera.data.angle_y
        # use euler angles to rotate about camera's x and y axes
        camera.rotation_mode = "XYZ"
        camera.rotation_euler.rotate_axis("Y", x_angle)
        camera.rotation_euler.rotate_axis("X", -y_angle)
        camera.rotation_mode = "QUATERNION"

        # set object pose, relative to camera
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = camera.rotation_quaternion @ self.pose

        # set lighting angle, relative to camera
        sun.rotation_mode = "QUATERNION"
        sun.rotation_quaternion = camera.rotation_quaternion @ self.lighting

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


# test code
for t in np.arange(0, 2 * np.pi + 0.001, 2 * np.pi / 100):
    # create a picture
    pic = Picture(position=(0, 0, 0),
                  distance=50,
                  pose=Spherical(10.5, 37, 0.4),
                  lighting=Spherical(0, 0, 0),
                  offset=(0.7, 0.2),
                  background=Spherical(t * 2, t, t)
                  )
    # set it up with cygnus, the camera, and the sun
    pic.setup(bpy.data.objects["Enhanced Cygnus"], bpy.data.objects["Camera"], bpy.data.objects["Sun"])
    # refresh view
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
