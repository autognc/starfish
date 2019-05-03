import numpy as np
from mathutils import Quaternion, Vector
from .utils import to_quat, jsonify
from .rotations import Spherical
import json

class Frame:
    """Represents a single picture of an object with certain parameters.

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
        """Initializes a picture with all of the parameters it needs.

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

    def dumps(self):
        return jsonify(self)

    def setup(self, obj, camera, sun):
        """Sets up a camera, object, and sun into the picture-taking position.

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
