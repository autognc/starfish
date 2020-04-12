import numpy as np
import cv2


def normalize_mask_colors(mask, colors, color_variation_cutoff=6):
    """
    Normalizes the colors of a mask image.

    Blender has a bug where the colors in a mask image vary slightly (e.g. instead of the background
    being solid rgb(0, 0, 0) black, it will actually be a random mix of rgb(0, 0, 1), rgb(1, 1, 0), etc...).
    This function takes a mask as well as a map of what the colors are supposed to be, then eliminates
    this variation.

    This function accepts either the path to the mask (str) or the mask itself represented as a numpy array. If a path
    is provided, then the function will return the normalized mask as well as overwrite the original mask on disk.
    If a numpy array is provided, then the function will just return the normalized mask.

    Args:
        :param mask: path to mask image (str) or numpy array of mask image (RGB)
        :param colors: a list of what the label colors are supposed to be, each in [R, G, B] format
        :param color_variation_cutoff:  colors will be allowed to differ from a color in the label map by a
            cityblock distance of no more than this value. The default value is 6, or equivalently 2 in each
            RGB channel. I chose this value because, in my experience with Blender 2.8,
            the color variation is no more than 1 in each channel, a number I then doubled to be safe.
        :return: the normalized mask as a numpy array
    """
    mask_path = None
    if isinstance(mask, str):
        mask_path = mask
        mask = cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB)

    colors = np.array(list(map(list, colors)))

    # shape (w, h, len(colors)) of cityblock distance from each pixel to each color
    distances = np.absolute(mask[:, :, None, :] - colors).sum(axis=3)
    mask = distances < color_variation_cutoff

    # check to make sure that every pixel belongs to exactly one label
    counts = np.count_nonzero(mask, axis=2)
    if np.any(counts > 1):
        raise ValueError(f'At least one pixel in {mask} belongs to more than one class')
    elif np.any(counts < 1):
        raise ValueError(f'At least one pixel in {mask} does not belong to a class')

    # perform replacement
    indices = np.argwhere(mask)[:, 2].reshape(mask.shape[:2])  # shape (w, h) of indices into colors
    result = colors[indices]

    result = result.astype(np.uint8)
    if mask_path:
        cv2.imwrite(mask_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return result
