import numpy as np
import cv2


def get_bounding_boxes_from_mask(mask, label_map):
    """Gets bounding boxes from instance masks.

    :param mask: path to mask image (str) or numpy array of mask image (RGB)
    :param label_map: dictionary mapping classes (str) to their corresponding color(s). Each class can correspond to a
        single color (e.g. ``{"cygnus": (0, 0, 206)}``) or multiple colors (e.g.
        ``{"cygnus": [(0, 0, 206), (206, 0, 0)]}``)

    :returns: a dictionary mapping classes (str) to their corresponding
        bboxes (a dictionary with the keys 'xmin', 'xmax', 'ymin', 'ymax'). If a class does not appear in the image,
        then it will not appear in the keys of the returned dictionary.
    """
    if isinstance(mask, str):
        mask = cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB)
    bboxes = {}
    for class_name, colors in label_map.items():
        colors = np.array(list(colors))
        # if a single color is provided, turn it into a list of length 1
        if len(colors.shape) == 1:
            colors = colors[None, ...]
        class_mask = np.any(np.all(mask[:, :, None, :] == colors, axis=-1), axis=-1)
        ys, xs = class_mask.nonzero()
        if len(ys) > 0:
            bboxes[class_name] = {
                'ymin': int(np.min(ys)),
                'ymax': int(np.max(ys)),
                'xmin': int(np.min(xs)),
                'xmax': int(np.max(xs)),
            }
    return bboxes


def get_centroids_from_mask(mask, label_map):
    """Gets centroids from instance masks.

    :param mask: path to mask image (str) or numpy array of mask image (RGB)
    :param label_map: dictionary mapping classes (str) to their corresponding color(s). Each class can correspond to a
        single color (e.g. ``{"cygnus": (0, 0, 206)}``) or multiple colors (e.g.
        ``{"cygnus": [(0, 0, 206), (206, 0, 0)]}``)

    :returns: a dictionary mapping classes (str) to their corresponding
        centroids (y, x). If a class does not appear in the image,
        then it will not appear in the keys of the returned dictionary.
    """
    if isinstance(mask, str):
        mask = cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB)
    centroids = {}
    for class_name, colors in label_map.items():
        colors = np.array(list(colors))
        # if a single color is provided, turn it into a list of length 1
        if len(colors.shape) == 1:
            colors = colors[None, ...]
        class_mask = np.any(np.all(mask[:, :, None, :] == colors, axis=-1), axis=-1)
        ys, xs = class_mask.nonzero()
        if len(ys) > 0:
            centroids[class_name] = (int(np.mean(ys)), int(np.mean(xs)))
    return centroids


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

    :param mask: path to mask image (str) or numpy array of mask image (RGB)
    :param colors: a list of what the label colors are supposed to be, each in [R, G, B] format
    :param color_variation_cutoff:  colors will be allowed to differ from a color in the label map by a
        cityblock distance of no more than this value. The default value is 6, or equivalently 2 in each
        RGB channel. I chose this value because, in my experience with Blender 2.8,
        the color variation is no more than 1 in each channel, a number I then doubled to be safe.
    :returns: the normalized mask as a numpy array
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
