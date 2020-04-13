import time
import bpy
import numpy as np
from mathutils import Euler
from starfish import Sequence
from starfish.utils import random_rotations
from starfish.postprocessing import normalize_mask_colors, get_centroids_from_mask, get_bounding_boxes_from_mask

# create a standard sequence of random configurations...
seq = Sequence.standard(
    pose=random_rotations(1000),
    lighting=random_rotations(1000),
    background=random_rotations(1000),
    distance=np.linspace(10, 50, num=1000)
)
# ...or an exhaustive sequence of combinations...
seq = Sequence.exhaustive(
    distance=[10, 20, 30],
    offset=[(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)],
    pose=[Euler((0, 0, 0)), Euler((np.pi, 0, 0))]
)
# ...or an interpolated sequence between keyframes...
seq = Sequence.interpolated(
    waypoints=Sequence.standard(distance=[10, 20], pose=[Euler((0, 0, 0)), Euler((0, np.pi, np.pi))]),
    counts=[100]
)

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
