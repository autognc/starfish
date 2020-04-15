import pytest
from starfish.annotation import get_bounding_boxes_from_mask, get_centroids_from_mask, normalize_mask_colors
import numpy as np


def test_bboxes():
    mask = np.zeros((1920, 1080, 3), dtype=np.uint8)
    mask[100:200, 100:200] = np.array([1, 2, 3])
    mask[100:200, 200:300] = np.array([4, 5, 6])

    label_map = {'box1': (1, 2, 3), 'box2': (4, 5, 6), 'doesnt_exist': (1, 1, 1)}
    assert get_bounding_boxes_from_mask(mask, label_map) == \
        {
            'box1': {'ymin': 100, 'ymax': 199, 'xmin': 100, 'xmax': 199},
            'box2': {'ymin': 100, 'ymax': 199, 'xmin': 200, 'xmax': 299},
        }
    assert get_centroids_from_mask(mask, label_map) == \
        {
            'box1': (149, 149),
            'box2': (149, 249)
        }

    label_map = {'box': [(1, 2, 3), (4, 5, 6)]}
    assert get_bounding_boxes_from_mask(mask, label_map) == \
        {
            'box': {'ymin': 100, 'ymax': 199, 'xmin': 100, 'xmax': 299},
        }
    assert get_centroids_from_mask(mask, label_map) == \
        {
            'box': (149, 199)
        }


def test_normalize_colors():
    clean_mask = np.full((1920, 1080, 3), 100, dtype=np.uint8)
    clean_mask[512:1024, 512:1024] = 200
    dirty_mask = clean_mask + np.random.randint(-1, 2, clean_mask.shape)

    assert np.all(normalize_mask_colors(dirty_mask, [(100, 100, 100), (200, 200, 200)]) == clean_mask)

    with pytest.raises(ValueError):
        clean_mask[512, 512, :] += np.array([2, 2, 3], dtype=np.uint8)
        normalize_mask_colors(clean_mask, [(100, 100, 100), (200, 200, 200)])

        normalize_mask_colors(dirty_mask, [(100, 100, 100), (101, 101, 101), (200, 200, 200)])