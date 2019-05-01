import numpy as np
import bpy
import ssi
from ssi.rotations import Spherical
from mathutils import Euler
from ssi import utils


def main():
    poses = [Euler(args) for args in utils.cartesian([0], np.linspace(0, np.pi, num=100), np.linspace(0, np.pi, num=10))]

    seq = ssi.Sequence.exhaustive(
        pose=poses,
        background=Spherical(0, 0, 0),
        distance=20
    )
    seq.bake(bpy.data.objects["Enhanced Cygnus"], bpy.data.objects["Camera"], bpy.data.objects["Sun"])

if __name__ == "__main__":
    main()
