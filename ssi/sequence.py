from . import Frame
import numpy as np

def interp(a, b, n, endpoint=True):
    """Interpolates between two frames.

    Args:
        a (ssi.Frame): the first frame to interpolate from
        b (ssi.Frame): the second frame to interpolate to
        n (int): the number of frames to generate
        endpoint (boolean): If True, frame b will be included in the result. Otherwise, it will be excluded. (default:
        True)

    Returns:
        List of ssi.Frame objects
    """
    # each key is an argument to the Frame constructor
    lists = {
        "position": np.array([np.linspace(x, y, n, endpoint) for x, y in zip(a.position, b.position)]).T,
        "distance": np.linspace(a.distance, b.distance, n, endpoint),
        "pose": [a.pose.slerp(b.pose, t) for t in np.linspace(0, 1, n, endpoint)],
        "lighting": [a.lighting.slerp(b.lighting, t) for t in np.linspace(0, 1, n, endpoint)],
        "offset": [t for t in zip(np.linspace(a.offset[0], b.offset[0], n, endpoint), np.linspace(a.offset[1], b.offset[1], n, endpoint))],
        "background": [a.background.to_quaternion().slerp(b.background.to_quaternion(), t) for t in np.linspace(0, 1, n, endpoint)]
    }

    # creates Frames out of the dict of lists
    frames = []
    for vals in zip(*lists.values()):
        kwargs = dict(zip(lists.keys(), vals))
        frames.append(Frame(**kwargs))

    return frames

class Sequence:
    def __init__(self, frames):
        self.frames = frames

    @classmethod
    def interpolated(this, waypoints, counts):
        try:
            iter(counts)
        except TypeError:
            counts = [counts]

        frames = []
        # add frames for all but last waypoint
        for a, b, n in zip(waypoints[:-1], waypoints[1:-1], counts[:-1]):
            frames += interp(a, b, n, endpoint=False)
        # include endpoint on the last waypoint
        frames += interp(waypoints[-2], waypoints[-1], counts[-1], endpoint=True)

        return this(frames)

    @classmethod
    def exhaustive(this, start, end):
        pass

    @classmethod
    def random(this, start, end):
        pass

    def setup(self, obj, camera, sun):
        """Sets up a camera, object, and sun into an animation of this sequence.

        Args:
            obj (BlendDataObject): the object that will be the subject of the picture
            camera (BlendDataObject): the camera to take the picture with
            sun (BlendDataObject): the sun lamp that is providing the lighting
        """

        for i, frame in enumerate(self.frames):
            frame.setup(obj, camera, sun)
            obj.keyframe_insert("location", frame=i + 1)
            obj.keyframe_insert("rotation_quaternion", frame=i + 1)
            camera.keyframe_insert("location", frame=i + 1)
            camera.keyframe_insert("rotation_quaternion", frame=i + 1)
            sun.keyframe_insert("rotation_quaternion", frame=i + 1)
