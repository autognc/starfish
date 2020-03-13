import numpy as np
from PIL import Image


def normalize_mask_colors(mask_path, colors, color_variation_cutoff=6):
    """
    Normalizes the colors of a mask image.

    Blender has a bug where the colors in a mask image vary slightly (e.g. instead of the background
    being solid rgb(0, 0, 0) black, it will actually be a random mix of rgb(0, 0, 1), rgb(1, 1, 0), etc...).
    This function takes a mask path as well as a map of what the colors are supposed to be, then eliminates
    this variation and saves a clean mask at the same path with the extension .norm.png.

    Args:
        :param mask_path: path to mask image (str)
        :param colors: a list of what the label colors are supposed to be, each in [R, G, B] format
        :param color_variation_cutoff:  colors will be allowed to differ from a color in the label map by a
            cityblock distance of no more than this value. The default value is 6, or equivalently 2 in each
            RGB channel. I chose this value because, in my experience with Blender 2.8,
            the color variation is no more than 1 in each channel, a number I then doubled to be safe.
    """
    colors = np.array(list(map(list, colors)))
    img = np.array(Image.open(mask_path))[..., :3]  # remove alpha channel

    # shape (w, h, len(colors)) of cityblock distance from each pixel to each color
    distances = np.absolute(img[:, :, None, :] - colors).sum(axis=3)
    mask = distances < color_variation_cutoff

    # check to make sure that every pixel belongs to exactly one label
    counts = np.count_nonzero(mask, axis=2)
    if np.any(counts > 1):
        raise ValueError(f'At least one pixel in {mask_path} belongs to more than one class')
    elif np.any(counts < 1):
        raise ValueError(f'At least one pixel in {mask_path} does not belong to a class')

    # perform replacement
    indices = np.argwhere(mask)[:, 2].reshape(mask.shape[:2])  # shape (w, h) of indices into colors
    result = colors[indices]

    Image.fromarray(result.astype(np.uint8)).save(mask_path)
