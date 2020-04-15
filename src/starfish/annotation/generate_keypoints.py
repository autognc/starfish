import heapq
import mathutils
import numpy as np

ALPHA = 8
BETA = 0.65
GAMMA = 1.5


def _sample_eliminate(points, target_num, volume):
    """Run sample elimination to get a Poisson disk distribution"""
    kdtree = mathutils.kdtree.KDTree(len(points))
    for i, point in enumerate(points):
        kdtree.insert(point, i)
    kdtree.balance()

    rmax = (volume / (4 * np.sqrt(2) * target_num)) ** (1 / 3)
    rmin = rmax * (1 - (target_num / len(points)) ** GAMMA) * BETA

    def weight(d):
        d_hat = min(d, 2 * rmax) if d > 2 * rmin else 2 * rmin
        return (1 - (d_hat / (2 * rmax))) ** ALPHA

    heap = [
        [-sum(weight(d) for _, _, d in kdtree.find_range(point, 2 * rmax)), i]
        for i, point in enumerate(points)
    ]
    # create a dict of pointers into the heap
    heap_dict = {i: heap[i] for i in range(len(heap))}
    heapq.heapify(heap)

    while len(heap_dict) > target_num:
        _, index = heapq.heappop(heap)
        if index == -1:
            continue
        del heap_dict[index]
        for _, i, d in kdtree.find_range(points[index], 2 * rmax):
            if i in heap_dict.keys():
                # mark old value as dirty
                heap_dict[i][1] = -1
                # insert new value
                heap_dict[i] = [heap_dict[i][0] + weight(d), i]
                heapq.heappush(heap, heap_dict[i])

    return [points[i] for i in heap_dict.keys()]


def _distribute_particles_random(obj, num, seed):
    """Distribute particles according to a uniform distribution over an object"""
    import bpy

    # randomly distribute keypoints using Blender's particle system
    modifier = obj.modifiers.new('', 'PARTICLE_SYSTEM')
    psys = modifier.particle_system
    psys.seed = seed
    psys.settings.count = num
    psys.settings.emit_from = 'FACE'
    psys.settings.distribution = 'RAND'
    psys.settings.use_even_distribution = True

    # force an update to distribute particles
    bpy.context.view_layer.update()
    eval_obj = bpy.context.evaluated_depsgraph_get().objects.get(obj.name, None)

    # extract particle locations and return
    particles = [tuple(obj.matrix_world.inverted() @ p.location)
                 for p in eval_obj.particle_systems[-1].particles.values()]
    obj.modifiers.remove(modifier)
    return particles


def generate_keypoints(obj, num, oversample=10, seed=0):
    """Generates evenly spaced 3D keypoints on the surface of an object.

    This function implements the Sample Elimination algorithm from
    `this paper <http://www.cemyuksel.com/research/sampleelimination/sampleelimination.pdf>`_ to generate points on the
    surface of the object that follow a `Poisson Disk <https://en.wikipedia.org/wiki/Supersampling#Poisson_disc>`_
    distribution. The Poisson Disk distribution guarantees that no two points are within a certain distance of each
    other in 3D space, ensuring that the keypoints are more spread out.

    The way Sample Elimination works is by first generating ``num * oversample`` points at random, and then eliminating
    points in a certain order until there are ``num`` left. Thus, a higher value of ``oversample`` will give more
    evenly spaced points.

    :param obj: (BlendDataObject): Blender object to operate on
    :param num: (int): number of points to generate
    :param oversample: (float): amount of oversampling to do (see above), default 10
    :param seed: (int): seed for the initial random point generation

    :return: a list of length ``num`` containing 3-tuples representing the coordinates of the keypoints in object space
    """
    import bmesh
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    volume = mesh.calc_volume()
    mesh.free()

    particles = _distribute_particles_random(obj, int(num * oversample), seed)
    return _sample_eliminate(particles, num, volume)
