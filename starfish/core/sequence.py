import numpy as np
from copy import deepcopy
from mathutils import Euler, Quaternion, Matrix, Vector

from starfish.utils import cartesian
from .frame import Frame


def interp(a, b, n, endpoint=True):
    """Interpolates between two frames.

    Args:
        a (starfish.Frame): the first frame to interpolate from
        b (starfish.Frame): the second frame to interpolate to
        n (int): the number of frames to generate
        endpoint (boolean): If True, frame b will be included in the result. Otherwise, it will be excluded. (default:
        True)

    Returns:
        List of starfish.Frame objects
    """
    # each key is an argument to the Frame constructor
    lists = {
        "position": np.array([np.linspace(x, y, n, endpoint) for x, y in zip(a.position, b.position)]).T,
        "distance": np.linspace(a.distance, b.distance, n, endpoint),
        "pose": [a.pose.slerp(b.pose, t) for t in np.linspace(0, 1, n, endpoint)],
        "lighting": [a.lighting.slerp(b.lighting, t) for t in np.linspace(0, 1, n, endpoint)],
        "offset": [t for t in zip(np.linspace(a.offset[0], b.offset[0], n, endpoint),
                                  np.linspace(a.offset[1], b.offset[1], n, endpoint))],
        "background": [a.background.to_quaternion().slerp(b.background.to_quaternion(), t) for t in
                       np.linspace(0, 1, n, endpoint)]
    }

    # creates Frames out of the dict of lists
    frames = []
    for vals in zip(*lists.values()):
        kwargs = dict(zip(lists.keys(), vals))
        frames.append(Frame(**kwargs))

    return frames


class Sequence:
    """
    Represents a sequence of frames.

    This class acts exactly like a list of starfish.Frame objects, with a few utility methods tacked on.

    Most of the power of this class comes from the classmethod constructors, which can be used to create different
    types of sequences in a more convenient, expressive way.

    The bake method is useful for previewing sequences in Blender while they are being tweaked and configured, in order
    to get an idea of what they will look like before rendering. However, it is not recommended to use bake at render
    time. Instead, the sequence object can be iterated over frame-by-frame, like so:

        for frame in sequence:
            frame.setup(...)
            bpy.ops.render.render(...)
    """

    def __init__(self, frames):
        """Initializes a sequence from a list of frames."""
        self.frames = frames

    @classmethod
    def standard(cls, **kwargs):
        """Creates a sequence from parameters that are lists.

        The arguments to this constructor are the same as those to the starfish.Frame constructor, except instead of
        a single value, each argument may also be a list of values. For example, while `position` is normally an
        iterable of length 3 representing a 3D vector, it could instead be a list of 3D vectors (e.g. an array of
        shape (n, 3)).

        This constructor then generates a list of frames where the parameters for each frame come from these lists,
        zipped together.

        Each list of parameters must be either the same length as all the others, or be a single value. If a single
        value is provided for a parameter, then that value is broadcasted across all the frames, i.e. every frame gets
        that value for that parameter. (The same thing happens if a parameter is omitted: every frame gets the default
        value for that parameter).
        """
        # if any parameter is a single value, turn it into a length-1 list
        kwargs = {k: cls._preprocess_arg(v) for k, v in kwargs.items()}

        kwargs_multi = {k: v for k, v in kwargs.items() if len(v) > 1}
        if kwargs_multi:
            lengths = list(map(len, kwargs_multi.values()))
            if not all(lengths[0] == l for l in lengths):
                raise ValueError('Parameter lists of differing lengths were provided')
            length = lengths[0]

            # perform broadcasting
            kwargs_single = {k: v for k, v in kwargs.items() if len(v) == 1}
            for k, v in kwargs_single.items():
                kwargs[k] = [deepcopy(v[0]) for _ in range(length)]

        # invert dict of lists to list of dicts
        kwargs_per_frame = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]
        return cls([Frame(**args) for args in kwargs_per_frame])

    @classmethod
    def interpolated(cls, waypoints, counts):
        """Creates a sequence interpolated from a list of waypoints.

        Args:
            waypoints (seq): A starfish.Sequence object (or just a list of starfish.Frame objects) representing the
                waypoints to interpolate between. (tip: use the starfish.Sequence.standard constructor to create this.)
            counts (int or seq): The number of frames to generate between each pair of waypoints. There will be
                counts[i] frames in between waypoints[i] (inclusive) and waypoints[i+1] (exclusive). The total number of
                frames in the sequence will be sum(counts) + 1. The length of counts should be 1 less than the length of
                waypoints. (If count is an integer, then there should only be 2 waypoints.)
        """
        try:
            iter(counts)
        except TypeError:
            counts = [counts]

        if len(counts) != len(waypoints) - 1:
            raise ValueError('The length of counts should be 1 less than the length of waypoints.')

        frames = []
        # add frames for all but last waypoint
        for a, b, n in zip(waypoints, waypoints[1:], counts):
            frames += interp(a, b, n, endpoint=False)
        # add endpoint
        frames.append(waypoints[-1])

        return cls(frames)

    @classmethod
    def exhaustive(cls, **kwargs):
        """Creates a sequence that includes every possible combination of the parameters given.

        The arguments to this constructor are the same as those to the starfish.Frame constructor, except instead of
        a single value, each argument may also be a list of values. For example, while `position` is normally an
        iterable of length 3 representing a 3D vector, it could instead be a list of 3D vectors (e.g. an array of
        shape (n, 3)).

        This constructor then takes the lists of values for each parameter and generates frames out of their cartesian
        product. For example, if 10 distances, 10 poses, and 10 offsets are provided, the generated sequence will be
        10*10*10 = 10,000 frames long, including every possible combination of given distances, poses, and offsets.
        """
        # if any parameter is a single value, turn it into a length-1 list
        kwargs = {k: cls._preprocess_arg(v) for k, v in kwargs.items()}

        # get cartesian product of all parameter lists
        combos = cartesian(*kwargs.values())
        # create a frame from each one and return
        frame_kwargs = [dict(zip(kwargs.keys(), combo)) for combo in combos]
        return cls([Frame(**args) for args in frame_kwargs])

    def bake(self, scene, obj, camera, sun, num=None):
        """
        Creates keyframes representing this sequence, so that it can be played as a preview animation.  Keyframes will
        be adjacent to each other, so no interpolation will be done. This is just a means to get an idea of what frames
        are in the sequence. If `len(frames)` is greater than `num`, only every `(len(frames) / num)` frames will be
        displayed.

        This should not be called with large values of `num` (>5000), as it is quite slow and may cause Blender to hang.

        Args:
            scene (BlendDataObject): the scene to set up the animation in
            obj (BlendDataObject): the object that will be the subject of the picture
            camera (BlendDataObject): the camera to take the picture with
            sun (BlendDataObject): the sun lamp that is providing the lighting
            num (int): The number of keyframes to generate. Defaults to min(100, len(frames))
        """
        if num is None:
            num = min(100, len(self.frames))

        obj.animation_data_clear()
        camera.animation_data_clear()
        sun.animation_data_clear()

        scene.frame_start = 1
        scene.frame_end = num

        for i, frame in enumerate(self.frames[::-(-len(self.frames) // num)]):
            frame.setup(scene, obj, camera, sun)
            obj.keyframe_insert("location", frame=i + 1)
            obj.keyframe_insert("rotation_quaternion", frame=i + 1)
            camera.keyframe_insert("location", frame=i + 1)
            camera.keyframe_insert("rotation_quaternion", frame=i + 1)
            sun.keyframe_insert("rotation_quaternion", frame=i + 1)

    @staticmethod
    def _preprocess_arg(arg):
        """Turn any non-iterable argument into a singleton list"""
        if type(arg) in [Euler, Quaternion, Matrix, Vector]:
            return [arg]
        try:
            iter(arg)
            return arg
        except TypeError:
            return [arg]

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, i):
        return self.frames[i]

    def __setitem__(self, i, v):
        self.frames[i] = v

    def __delitem__(self, key):
        del self.frames[key]
