"""
IMPORTANT NOTE: This script is just for demonstrating the various capabilities of Starfish, and is not meant
to be run as-is. If you try to run this script without modifications, it will probably not work, unless you have
your Blend file set up with the exact same scenes, objects, and compositing nodes. Even then, it will immediately
start rendering several long sequences and writing files to disk, with the files from each sequence overwriting
the files from the previous one.
"""
import time
import bpy
import numpy as np
from mathutils import Euler
from starfish import Sequence
from starfish.utils import random_rotations
from starfish.annotation import normalize_mask_colors, get_centroids_from_mask, get_bounding_boxes_from_mask

# create a standard sequence of random configurations...
seq1 = Sequence.standard(
    pose=random_rotations(100),
    lighting=random_rotations(100),
    background=random_rotations(100),
    distance=np.linspace(10, 50, num=100)
)
# ...or an exhaustive sequence of combinations...
seq2 = Sequence.exhaustive(
    distance=[10, 20, 30],
    offset=[(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)],
    pose=[Euler((0, 0, 0)), Euler((np.pi, 0, 0))]
)
# ...or an interpolated sequence between keyframes...
seq3 = Sequence.interpolated(
    waypoints=Sequence.standard(distance=[10, 20], pose=[Euler((0, 0, 0)), Euler((0, np.pi, np.pi))]),
    counts=[100]
)

for seq in [seq1, seq2, seq3]:
    # render loop
    for i, frame in enumerate(seq):
        # non-starfish Blender stuff: e.g. setting file output paths
        bpy.data.scenes['Real'].node_tree.nodes['File Output'].file_slots[0].path = f'real_{i}.png'
        bpy.data.scenes['Mask'].node_tree.nodes['File Output'].file_slots[0].path = f'mask_{i}.png'

        # set up and render
        scene = bpy.data.scenes['Real']
        frame.setup(scene, bpy.data.objects['MyObject'],
                    bpy.data.objects['MyCamera'], bpy.data.objects['TheSun'])
        bpy.ops.render.render(scene=scene)

        scene = bpy.data.scenes['Mask']
        frame.setup(scene, bpy.data.objects['MyObject'],
                    bpy.data.objects['MyCamera'], bpy.data.objects['TheSun'])
        bpy.ops.render.render(scene=scene)

        # postprocessing
        label_map = {'object': (255, 255, 255), 'background': (0, 0, 0)}
        clean_mask = normalize_mask_colors(f'mask_{i}.png', label_map.values())
        del label_map['background']
        bboxes = get_bounding_boxes_from_mask(clean_mask, label_map)
        centroids = get_centroids_from_mask(clean_mask, label_map)

        # add some extra metadata
        frame.timestamp = int(time.time() * 1000)
        frame.sequence_name = '1000 random poses'
        frame.tags = ['front_view', 'left_view', 'right_view']
        frame.bboxes = bboxes
        frame.centroids = centroids

        # save metadata to JSON
        with open(f'meta_{i}.json', 'w') as f:
            f.write(frame.dumps())
