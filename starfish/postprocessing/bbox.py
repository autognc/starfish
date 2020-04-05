import numpy as np
import cv2


def get_bounding_boxes_from_mask(mask, label_map):
    """
    Gets bounding boxes from instance masks.
    :param mask: path to mask image (str) or numpy array of mask image (RGB)
    :param label_map: dictionary mapping classes (str) to their corresponding color(s). Each class can correspond to a
        single color (e.g. {"cygnus": (0, 0, 206)}) or multiple colors (e.g. {"cygnus": [(0, 0, 206), (206, 0, 0)]})
    :return: a dictionary mapping classes (str) to their corresponding
        bboxes (a dictionary with the keys 'xmin', 'xmax', 'ymin', 'ymax'). If a class does not appear in the image,
        then it will not appear in the keys of the returned dictionary.
    """
    if type(mask) is not np.ndarray:
        mask = cv2.imread(mask)
    bboxes = {}
    for class_name, colors in label_map.items():
        colors = np.array(list(colors))
        # if a single color is provided, turn it into a list of length 1
        if len(colors.shape) == 1:
            colors = [None, ...]
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
    """
    Gets bounding boxes from instance masks.
    :param mask: path to mask image (str) or numpy array of mask image (RGB)
    :param label_map: dictionary mapping classes (str) to their corresponding color(s). Each class can correspond to a
        single color (e.g. {"cygnus": (0, 0, 206)}) or multiple colors (e.g. {"cygnus": [(0, 0, 206), (206, 0, 0)]})
    :return: a dictionary mapping classes (str) to their corresponding
        centroids (y, x). If a class does not appear in the image,
        then it will not appear in the keys of the returned dictionary.
    """
    if type(mask) is not np.ndarray:
        mask = cv2.imread(mask)
    centroids = {}
    for class_name, colors in label_map.items():
        colors = np.array(list(colors))
        # if a single color is provided, turn it into a list of length 1
        if len(colors.shape) == 1:
            colors = [None, ...]
        class_mask = np.any(np.all(mask[:, :, None, :] == colors, axis=-1), axis=-1)
        ys, xs = class_mask.nonzero()
        if len(ys) > 0:
            centroids[class_name] = (int(np.mean(ys)), int(np.mean(xs)))
    return centroids
