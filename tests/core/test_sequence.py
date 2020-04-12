import pytest
from starfish import Sequence, Frame
from starfish.rotations import Spherical
from mathutils import Quaternion, Vector


class TestSequence:
    @staticmethod
    def sequence_equal(a, b):
        return all(vars(a1) == vars(b1) for a1, b1 in zip(a, b))

    def test_standard(self):
        # one-frame sequence
        assert self.sequence_equal(
            Sequence.standard(),
            [Frame()]
        )
        # vary distance while all others all default
        assert self.sequence_equal(
            Sequence.standard(distance=[1, 2, 3]),
            [Frame(distance=1), Frame(distance=2), Frame(distance=3)]
        )
        # 1-frame sequence with all parameters specified
        assert self.sequence_equal(
            Sequence.standard(
                position=[(0, 0, 0)],
                distance=[1],
                pose=[Quaternion()],
                lighting=[Quaternion([1, 1, 1, 1])],
                offset=[(0.5, 0.5)],
                background=[Quaternion([1, 1, 1, 1])],
            ),
            [
                Frame(position=(0, 0, 0), distance=1, pose=Quaternion(),
                      lighting=Quaternion([1, 1, 1, 1]), offset=(0.5, 0.5), background=Quaternion([1, 1, 1, 1]))
            ]
        )
        # 2-frame sequence with all parameters specified
        assert self.sequence_equal(
            Sequence.standard(
                position=[(0, 0, 0), (1, 1, 1)],
                distance=[1, 2],
                pose=[Quaternion(), Quaternion([1, 1, 1, 1])],
                lighting=[Quaternion([1, 1, 1, 1]), Quaternion()],
                offset=[(0.5, 0.5), (1, 1)],
                background=[Quaternion([1, 1, 1, 1]), Quaternion([1, 1, 1, 1])]
            ),
            [
                Frame(position=(0, 0, 0), distance=1, pose=Quaternion(),
                      lighting=Quaternion([1, 1, 1, 1]), offset=(0.5, 0.5), background=Quaternion([1, 1, 1, 1])),
                Frame(position=(1, 1, 1), distance=2, pose=Quaternion([1, 1, 1, 1]),
                      lighting=Quaternion(), offset=(1, 1), background=Quaternion([1, 1, 1, 1])),
            ]
        )
        # 3-frame sequence with broadcasting on all parameters except position
        assert self.sequence_equal(
            Sequence.standard(
                position=[(0, 0, 0), (1, 1, 1), (2, 2, 2)],
                distance=[1],
                pose=[Quaternion([1, 1, 1, 1])],
                lighting=[Quaternion([1, 1, 1, 1])],
                offset=[(1, 1)],
                background=[Quaternion([1, 1, 1, 1])]
            ),
            [
                Frame(position=(0, 0, 0), distance=1, pose=Quaternion([1, 1, 1, 1]),
                      lighting=Quaternion([1, 1, 1, 1]), offset=(1, 1), background=Quaternion([1, 1, 1, 1])),
                Frame(position=(1, 1, 1), distance=1, pose=Quaternion([1, 1, 1, 1]),
                      lighting=Quaternion([1, 1, 1, 1]), offset=(1, 1), background=Quaternion([1, 1, 1, 1])),
                Frame(position=(2, 2, 2), distance=1, pose=Quaternion([1, 1, 1, 1]),
                      lighting=Quaternion([1, 1, 1, 1]), offset=(1, 1), background=Quaternion([1, 1, 1, 1])),
            ]
        )
        with pytest.raises(ValueError):
            Sequence.standard(distance=[1, 2, 3], position=[(0, 0, 0), (0, 0, 0)])
            Sequence.standard(distance=[1, 2, 3], position=(0, 0, 0))
            Sequence.standard(distance=[1, 2, 3, 4], background=Quaternion())
            Sequence.standard(distance=(1, 2, 3))

    def test_interpolated(self):
        seq = Sequence.standard(distance=[1, 2, 3])
        counts = [10, 10]
        interpolated = Sequence.interpolated(seq, counts)

        frames = [Frame(distance=1), Frame(distance=2), Frame(distance=3)]
        assert self.sequence_equal(interpolated, Sequence.interpolated(frames, counts))

        assert len(interpolated) == sum(counts) + 1
        assert vars(interpolated[-1]) == vars(seq[-1])

        seq = Sequence.standard(distance=[1, 2])
        assert self.sequence_equal(Sequence.interpolated(seq, 10), Sequence.interpolated(seq, [10]))

        with pytest.raises(ValueError):
            Sequence.interpolated(seq, [10, 10])
            Sequence.interpolated(frames, 10)
            Sequence.interpolated(frames, [10])
            Sequence.interpolated(frames, [10, 10, 20])

    @staticmethod
    def sequence_set_equal(a, b):
        """Tests sequence equality where order doesn't matter"""
        def freeze(value):
            if type(value) in [Vector, Quaternion]:
                return tuple(value)
            if type(value) is Spherical:
                return value.theta, value.phi, value.roll
            return value

        set_a = set(frozenset((k, freeze(v)) for k, v in vars(a1).items()) for a1 in a)
        set_b = set(frozenset((k, freeze(v)) for k, v in vars(b1).items()) for b1 in b)
        return set_a == set_b

    def test_exhaustive(self):
        assert self.sequence_set_equal(
            Sequence.exhaustive(),
            [Frame()]
        )
        assert self.sequence_set_equal(
            Sequence.exhaustive(distance=[1, 2], position=[(0, 0, 0), (1, 1, 2)],
                                pose=[Quaternion(), Quaternion([1, 1, 1, 1])]),
            [
                Frame(distance=1, position=(0, 0, 0), pose=Quaternion()),
                Frame(distance=1, position=(0, 0, 0), pose=Quaternion([1, 1, 1, 1])),
                Frame(distance=1, position=(1, 1, 2), pose=Quaternion()),
                Frame(distance=1, position=(1, 1, 2), pose=Quaternion([1, 1, 1, 1])),
                Frame(distance=2, position=(0, 0, 0), pose=Quaternion()),
                Frame(distance=2, position=(0, 0, 0), pose=Quaternion([1, 1, 1, 1])),
                Frame(distance=2, position=(1, 1, 2), pose=Quaternion()),
                Frame(distance=2, position=(1, 1, 2), pose=Quaternion([1, 1, 1, 1])),
                Frame(distance=2, position=(1, 1, 2), pose=Quaternion([1, 1, 1, 1])),
            ]
        )

