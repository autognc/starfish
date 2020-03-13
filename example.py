import bpy
import starfish
from starfish import utils
import time
import os


def main():
    poses = utils.random_rotations(1000)

    seq = starfish.Sequence.exhaustive(
        pose=poses,
        distance=20
    )

    output_node = bpy.data.scenes["Render"].node_tree.nodes["File Output"]
    output_node.base_path = "/home/black/TSL/render"
    for i, frame in enumerate(seq):
        frame.setup(bpy.data.scenes['Real'], bpy.data.objects["Enhanced Cygnus"], bpy.data.objects["Camera"], bpy.data.objects["Sun"])

        # add metadata to frame
        frame.timestamp = int(time.time() * 1000)
        frame.sequence_name = "1000 random poses"

        # set output path
        output_node.file_slots[0].path = f"real#_{i}"
        output_node.file_slots[1].path = f"mask#_{i}"

        # dump data to json
        with open(os.path.join(output_node.base_path, f"{i}.json"), "w") as f:
            f.write(frame.dumps())

        # render
        bpy.ops.render.render(scene="Render")


if __name__ == "__main__":
    main()
