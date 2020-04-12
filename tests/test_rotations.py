import pytest
from starfish.rotations import Spherical
import numpy as np
from mathutils import Quaternion, Euler, Matrix


class TestSpherical:
    @staticmethod
    def geodesic_distance(a, b):
        return 2 * np.arccos(np.clip(np.abs(np.sum(np.array(a) * np.array(b))), 0, 1))

    @staticmethod
    def random_spherical():
        return Spherical(*(np.random.random(3) * 2 * np.pi))

    def test_zero_rotation(self):
        zero_spherical = Spherical(0, 0, 0)
        zero_quaternion = Quaternion([1, 0, 0, 0])
        zero_euler = Euler([0, 0, 0])
        zero_matrix = Matrix.Identity(3)

        assert zero_spherical.to_quaternion() == zero_quaternion
        assert Spherical.from_other(zero_quaternion) == zero_spherical
        assert Spherical.from_other(zero_euler) == zero_spherical
        assert Spherical.from_other(zero_matrix) == zero_spherical

    @pytest.mark.repeat(100)
    def test_random_conversions(self):
        quat = self.random_spherical().to_quaternion()
        assert self.geodesic_distance(
            quat,
            Spherical.from_other(quat).to_quaternion()
        ) < 0.01

    def test_inputs_greater_than_2_pi_or_less_than_0(self):
        assert Spherical(2 * np.pi, 2 * np.pi, 2 * np.pi) == Spherical(0, 0, 0)
        assert Spherical(-2 * np.pi, -2 * np.pi, -2 * np.pi) == Spherical(0, 0, 0)
        assert Spherical(7 * np.pi, 7 * np.pi, 7 * np.pi) == Spherical(np.pi, np.pi, np.pi)
        assert Spherical(-7 * np.pi, -7 * np.pi, -7 * np.pi) == Spherical(np.pi, np.pi, np.pi)
