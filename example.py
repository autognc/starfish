import numpy as np
import bpy
import ssi
from ssi.rotations import Spherical

def main():
    # test code
    for t in np.arange(0, 2 * np.pi + 0.001, 2 * np.pi / 100):
        # create a picture
        frame = ssi.Frame(
            position=(0, 0, 0),
            distance=50,
            pose=Spherical(10.5, 37, 0.4),
            lighting=Spherical(0, 0, 0),
            offset=(0.7, 0.2),
            background=Spherical(t * 2, t, t)
        )
        # set it up with cygnus, the camera, and the sun
        frame.setup(bpy.data.objects["Enhanced Cygnus"], bpy.data.objects["Camera"], bpy.data.objects["Sun"])
        # refresh view
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

if __name__ == "__main__":
    main()
