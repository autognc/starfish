import heapq
import mathutils
import numpy as np

ALPHA = 8
BETA = 0.65
GAMMA = 1.5


def _compute_rmax_rmin(curr_num, target_num, volume):
    rmax = (volume / (4 * np.sqrt(2) * target_num)) ** (1 / 3)
    rmin = rmax * (1 - (target_num / curr_num) ** GAMMA) * BETA
    return rmax, rmin


def _weight(d, rmax, rmin):
    d_hat = min(d, 2 * rmax) if d > 2 * rmin else 2 * rmin
    return (1 - (d_hat / (2 * rmax))) ** ALPHA


def _sample_eliminate(points, target_num, stop_num, volume):
    """Run sample elimination to get a Poisson disk distribution"""
    kdtree = mathutils.kdtree.KDTree(len(points))
    for i, point in enumerate(points):
        kdtree.insert(point, i)
    kdtree.balance()

    rmax, rmin = _compute_rmax_rmin(len(points), target_num, volume)
    heap = [
        [
            -sum(_weight(d, rmax, rmin) for _, _, d in kdtree.find_range(point, 2 * rmax)),
            i
        ]
        for i, point in enumerate(points)
    ]
    # create a dict that maps from index to pointers into the heap
    heap_dict = {e[1]: e for e in heap}
    heapq.heapify(heap)

    result_indices = []
    curr_target = target_num
    while len(heap_dict) > stop_num:
        if len(heap_dict) == curr_target:
            # move down target size by factors of 2, as in the paper
            curr_target //= 2
            # update rmax, rmin, and heap values
            rmax, rmin = _compute_rmax_rmin(len(heap_dict), curr_target, volume)
            heap = [
                [
                    -sum(_weight(d, rmax, rmin) for _, ni, d in kdtree.find_range(points[i], 2 * rmax)
                         if ni in heap_dict.keys()),
                    i
                ]
                for i in heap_dict.keys()
            ]
            heap_dict = {e[1]: e for e in heap}
            heapq.heapify(heap)

        _, index = heapq.heappop(heap)
        if index == -1:
            continue

        if len(heap_dict) <= target_num:
            # we've reached the original target, so we need to start keeping track of the ordering
            result_indices.append(index)

        del heap_dict[index]

        # update points adjacent to the one that was just removed
        for _, ni, d in kdtree.find_range(points[index], 2 * rmax):
            if ni in heap_dict.keys():
                # mark old value as dirty
                heap_dict[ni][1] = -1
                # insert new value
                heap_dict[ni] = [heap_dict[ni][0] + _weight(d, rmax, rmin), ni]
                heapq.heappush(heap, heap_dict[ni])

    # reverse indices and then add the rest in no particular order
    result_indices = result_indices[::-1] + list(heap_dict.keys())

    return [points[i] for i in result_indices]


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
    if len(particles) != num:
        raise RuntimeError('Error generating initial random particles')
    return particles


def generate_keypoints(obj, num, stop=1, oversample=10, seed=0):
    """Generates evenly spaced 3D keypoints on the surface of an object.

    This function implements the Sample Elimination algorithm from
    `this paper <http://www.cemyuksel.com/research/sampleelimination/sampleelimination.pdf>`_ to generate points on the
    surface of the object that follow a `Poisson Disk <https://en.wikipedia.org/wiki/Supersampling#Poisson_disc>`_
    distribution. The Poisson Disk distribution guarantees that no two points are within a certain distance of each
    other in 3D space, ensuring that the keypoints are more spread out.

    The way Sample Elimination works is by first generating ``num * oversample`` points at random, and then eliminating
    points in a certain order until there are ``num`` left. Thus, a higher value of ``oversample`` will give more
    evenly spaced points.

    This also has the nice property that every intermediary set of points also follows a Poisson Disk distribution.
    By default, this function will keep running sample elimination until there is 1 point left, and then return the
    points in reverse order of elimination so that the first ``n`` points are also evenly spaced out for any ``1 <= n
    <= num``. The point at which Sample Elimination stops can be controlled with the ``stop`` parameter.

    :param obj: (BlendDataObject): Blender object to operate on
    :param num: (int): number of points to generate
    :param stop: (int): an integer between 1 and ``num`` (inclusive) at which sample elimination will stop, default 1
    :param oversample: (float): amount of oversampling to do (see above), default 10
    :param seed: (int): seed for the initial random point generation

    :return: A list of length ``num`` containing 3-tuples representing the coordinates of the keypoints in object space.
        The first ``n`` elements of the list will also be evenly spaced out for any ``stop <= n <= num``.
    """
    if stop < 1 or stop > num:
        raise ValueError('stop must be between 1 and num, inclusive')
    if oversample < 1:
        raise ValueError('oversample must be greater than or equal to 1')

    import bmesh
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    volume = mesh.calc_volume()
    mesh.free()

    particles = _distribute_particles_random(obj, int(num * oversample), seed)
    return _sample_eliminate(particles, num, stop, volume)
