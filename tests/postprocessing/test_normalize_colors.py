import pytest
from starfish.postprocessing import normalize_mask_colors
import numpy as np


def test_normalize_colors():
    clean_mask = np.full((1920, 1080, 3), 100, dtype=np.uint8)
    clean_mask[512:1024, 512:1024] = 200
    dirty_mask = clean_mask + np.random.randint(-1, 2, clean_mask.shape)

    assert np.all(normalize_mask_colors(dirty_mask, [(100, 100, 100), (200, 200, 200)]) == clean_mask)

    with pytest.raises(ValueError):
        clean_mask[512, 512, :] += np.array([2, 2, 3], dtype=np.uint8)
        normalize_mask_colors(clean_mask, [(100, 100, 100), (200, 200, 200)])

        normalize_mask_colors(dirty_mask, [(100, 100, 100), (101, 101, 101), (200, 200, 200)])
