import sys
import starfish
import starfish.annotation


class Dummy:
    def __setattr__(self, *_, **__):
        return Dummy()

    def __getattr__(self, *_, **__):
        return Dummy()

    def __getitem__(self, *_, **__):
        return Dummy()

    def __setitem__(self, *_, **__):
        return Dummy()

    def __call__(self, *_, **__):
        return Dummy()

    def __enter__(self, *_, **__):
        return Dummy()

    def __exit__(self, *_, **__):
        return Dummy()


def test_example(monkeypatch):
    monkeypatch.setitem(sys.modules, 'bpy', Dummy())
    monkeypatch.setattr(starfish.annotation, 'normalize_mask_colors', Dummy())
    monkeypatch.setattr(starfish.annotation, 'get_bounding_boxes_from_mask', Dummy())
    monkeypatch.setattr(starfish.annotation, 'get_centroids_from_mask', Dummy())
    monkeypatch.setattr(starfish.Frame, 'setup', Dummy())
    monkeypatch.setattr(starfish.Frame, 'dumps', Dummy())
    with open('example.py', 'r') as f:
        example = f.read()
    monkeypatch.setitem(__builtins__, 'open', Dummy())
    exec(example)
