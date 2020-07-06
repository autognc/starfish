import numpy as np
from mathutils import Quaternion, Vector, Euler

from starfish.rotations import Spherical
from starfish.utils import to_quat, jsonify


class Frame:
    """Represents a single picture of an object with certain parameters.

    There are 6 parameters that independently define a picture:

    * Object position: the absolute 3D position of the object in the global coordinate system (i.e. where it is in
      the scene).
    * Camera distance: the distance of the camera from the object.
    * Object pose: the pose of the object relative to the camera (i.e. how it will appear to be oriented in the
      picture).
    * Lighting: the angle from which the sun's rays will hit the object in the picture (e.g. from above, from the
      right, from behind the camera, etc.).
    * Object offset: the 2D translational offset of the object from the center of the picture frame.
    * Background/camera orientation: the orientation of the camera and object relative to the global coordinate
      system. This affects only what part of the scene appears in the background directly behind the object.

    A note on coordinate systems: the representations for `pose` and `translation` were carefully chosen
    to match those of OpenCV rather than using Blender's default coordinate system. This means that OpenCV camera
    projection functions such as ``projectPoints`` and ``solvePnP`` should produce correct results when `pose`
    and `translation` are treated as the ``rvec`` and ``tvec``, respectively.

    """

    def __init__(self, *, position=(0, 0, 0), distance=100, pose=Quaternion(),
                 lighting=Quaternion(), offset=(0.5, 0.5), background=Quaternion()):
        """Initializes a picture with all of the parameters it needs.

        A type of "rotation" means a `mathutils.Quaternion` object or any object with a to_quaternion() method (which
        includes `mathutils.Euler`, `mathutils.Matrix`, and `starfish.rotations.Spherical`).

        :param position: (seq, len 3): the (x, y, z) absolute position of the object in the scene's global coordinate
            system (default: (0, 0, 0))
        :param distance: (float or int): the distance of the camera from the object in blender units (default: 100)
        :param pose: (rotation): the orientation of the object relative to the camera's coordinate system (default: the
            identity quaternion (aka zero rotation), which corresponds to the camera looking directly in the
            object's +Z direction with the object's +X direction pointing to the right and +Y pointing down)
        :param lighting: (rotation): the angle of the sun's lighting relative to the camera's coordinate system (
            default: the identity quaternion (aka zero rotation), which corresponds to the light coming from directly
            behind the camera)
        :param offset: (seq of float, len 2): the (vertical, horizontal) translational offset of the object from the
            center of the picture frame. Expressed as a fraction of the distance from edge to edge: e.g., for horizontal
            offset, 0.0 is the left edge, 0.5 is the center, and 1.0 is the right edge. Same for vertical,
            but 0.0 is the top edge and 1.0 is the bottom. (default: (0.5, 0.5))
        :param background: (rotation): Imagine a ray starting at the camera and passing through the object. This
            parameter determines the orientation of this ray in the global coordinate system. For example, if you have a
            world background image that encircles your entire scene, two degrees of freedom of this parameter will
            determine the point in the background image that will appear directly behind the object,
            and the third degree of freedom will determine the rotation of this background image (i.e. which way is
            'up'). (default: the identity quaternion (aka zero rotation), which corresponds to the
            camera->object ray pointing directly in the -Z direction with the +X direction pointing to the right and +Y
            pointing up)
        """
        self.position = Vector(position)
        self.distance = distance
        self.pose = to_quat(pose)
        self.lighting = to_quat(lighting)
        self.offset = tuple(offset)
        self.background = to_quat(background)

        self.translation = None
        """The object's position relative to the camera represented by a single translation vector (in
        Blender units). This value isn't computed until setup time, and will be ``None`` beforehand. If you need this
        translation vector as part of your metadata, make sure to call `setup` first before calling `dumps`."""

    def dumps(self):
        """
        Converts all of the frame's attributes to a JSON object. By default, this will be the 6 frame parameters, plus
        `translation` if `setup` has already been called. Any additional metadata can be added by just setting it as
        an attribute: e.g. ``frame.sequence_name = '20k_square_earth_background'; metadata = frame.dumps()``
        """
        return jsonify(self)

    def setup(self, scene, obj, camera, sun):
        """Sets up a camera, object, and sun into the picture-taking position. Also computes and stores the translation
        vector of the object.

        :param scene: (BlendDataObject): the scene to use for aspect ratio calculations. Note that this should be the
            scene that you intend to perform the final render in, not necessarily the one that your objects exist in. If
            you render in a scene that has an output resolution with a different aspect ratio than the output
            resolution of this scene, then the offset of the object may be incorrect.
        :param obj: (BlendDataObject): the object that will be the subject of the picture
        :param camera: (BlendDataObject): the camera to take the picture with
        :param sun: (BlendDataObject): the sun lamp that is providing the lighting
        """
        # set object position
        obj.location = self.position

        # set camera position
        bg = Spherical.from_other(self.background)
        cartesian = self.distance * Vector((np.sin(bg.phi) * np.cos(bg.theta),
                                            np.sin(bg.phi) * np.sin(bg.theta),
                                            np.cos(bg.phi)))
        camera.location = cartesian + self.position

        # calculate camera angle offset to get correct object offset
        y_frac, x_frac = self.offset
        # convert proportions to angle offsets using camera frame. Taken partially
        # from bpy_extras.object_utils.world_to_camera_view. After inspecting the source code,
        # the `view_frame` method seems to only need the `scene` argument to compute the output
        # aspect ratio, hence the warning in this method's docstring.
        view_frame = camera.data.view_frame(scene=scene)[0]
        x_offset = (x_frac - 0.5) * 2 * view_frame.x
        y_offset = (y_frac - 0.5) * 2 * view_frame.y
        x_angle = np.arctan2(x_offset, -view_frame.z)
        y_angle = np.arctan2(y_offset, -view_frame.z)
        angle_offset = Euler([y_angle, x_angle, 0]).to_quaternion()

        # set camera rotation
        camera.rotation_mode = "QUATERNION"
        camera.rotation_quaternion = self.background @ angle_offset

        # set lighting angle, relative to camera
        sun.rotation_mode = "QUATERNION"
        sun.rotation_quaternion = camera.rotation_quaternion @ self.lighting

        # set object pose, relative to camera
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = camera.rotation_quaternion @ Euler([np.pi, 0, 0]).to_quaternion() @ self.pose

        # record translation vector from camera to object reference frame
        self.translation = camera.rotation_quaternion.inverted() @ (camera.location - self.position)
        self.translation.x = -self.translation.x

        # update world matrices in case they need to be used before next render
        obj.matrix_world = obj.matrix_basis
        camera.matrix_world = camera.matrix_basis
        sun.matrix_world = sun.matrix_basis
