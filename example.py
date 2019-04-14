import numpy as np
import bpy
import ssi
from ssi.rotations import Spherical
from mathutils import Euler


def main():
    # create two frames
    a = ssi.Frame(
        pose=Euler((0, 0, 0)),
        background=Spherical(0, 0, 0),
        distance=20
    )
    b = ssi.Frame(
        pose=Euler((np.pi, 0, 0)),
        background=Spherical(np.pi, np.pi, np.pi),
        distance=20
    )

    # create interpolated sequence
    seq = ssi.Sequence.interpolated([a, b], 100)
    # set it up with cygnus, the camera, and the sun
    seq.setup(bpy.data.objects["Enhanced Cygnus"], bpy.data.objects["Camera"], bpy.data.objects["Sun"])

if __name__ == "__main__":
    main()
