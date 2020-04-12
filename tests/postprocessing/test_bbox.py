from starfish.postprocessing import get_bounding_boxes_from_mask, get_centroids_from_mask
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
