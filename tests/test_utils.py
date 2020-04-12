from starfish import utils
from types import SimpleNamespace
from mathutils import Vector, Quaternion, Euler, Matrix
import json


def depth_2_all_equal(a, b):
    return all(all(a1 == b1 for a1, b1 in zip(a2, b2)) for a2, b2 in zip(a, b))


def test_cartesian():
    assert depth_2_all_equal(
        utils.cartesian([0], [1, 2], [3, 4, 5]),
        [[0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 2, 3], [0, 2, 4], [0, 2, 5]]
    )
    assert depth_2_all_equal(
        utils.cartesian([0], [1], [2]),
        [[0, 1, 2]]
    )
    assert depth_2_all_equal(
        utils.cartesian([0], ['a', 'b'], [None]),
        [[0, 'a', None], [0, 'b', None]]
    )
    assert depth_2_all_equal(
        utils.cartesian([Vector((1, 2, 3))], [Vector((4, 5, 6)), Vector((7, 8, 9))]),
        [[Vector((1, 2, 3)), Vector((4, 5, 6))], [Vector((1, 2, 3)), Vector((7, 8, 9))]]
    )


def test_jsonify():
    attrs = {
        'none': None,
        'string': 'string',
        'int': 0,
        'vector': Vector([1, 2, 3]),
        'rotation_quaternion': Quaternion([1, 0, 0, 0]),
        'rotation_matrix': Matrix.Identity(3),
        'rotation_euler': Euler([0, 0, 0]),
        'simple_list': [0, 1, 2, 'asdf', Euler([0, 0, 0]), Vector([1, 2, 3])],
        'nested_list': [0, 1, [2, 'asdf', [Euler([0, 0, 0]), Vector([1, 2, 3])]]],
        'nested_dict': {
            'none': None,
            'string': 'string',
            'int': 0,
            'vector': Vector([1, 2, 3]),
            'rotation_quaternion': Quaternion([1, 0, 0, 0]),
            'rotation_matrix': Matrix.Identity(3),
            'rotation_euler': Euler([0, 0, 0]),
            'simple_list': [0, 1, 2, 'asdf', Euler([0, 0, 0]), Vector([1, 2, 3])],
            'nested_list': [0, 1, [2, 'asdf', [Euler([0, 0, 0]), Vector([1, 2, 3])]]]
        }
    }
    expected = {
        'none': None,
        'string': 'string',
        'int': 0,
        'vector': [1, 2, 3],
        'rotation_quaternion': [1, 0, 0, 0],
        'rotation_matrix': [1, 0, 0, 0],
        'rotation_euler': [1, 0, 0, 0],
        'simple_list': [0, 1, 2, 'asdf', [1, 0, 0, 0], [1, 2, 3]],
        'nested_list': [0, 1, [2, 'asdf', [[1, 0, 0, 0], [1, 2, 3]]]],
        'nested_dict': {
            'none': None,
            'string': 'string',
            'int': 0,
            'vector': [1, 2, 3],
            'rotation_quaternion': [1, 0, 0, 0],
            'rotation_matrix': [1, 0, 0, 0],
            'rotation_euler': [1, 0, 0, 0],
            'simple_list': [0, 1, 2, 'asdf', [1, 0, 0, 0], [1, 2, 3]],
            'nested_list': [0, 1, [2, 'asdf', [[1, 0, 0, 0], [1, 2, 3]]]]
        }
    }

    assert expected == json.loads(utils.jsonify(SimpleNamespace(**attrs)))
