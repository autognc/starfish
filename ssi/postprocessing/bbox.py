import numpy as np
from PIL import Image


def get_bounding_boxes_from_mask(mask_path, label_map):
    """
    Gets bounding boxes from instance masks.
    :param mask_path: path to mask image (str)
    :param label_map: dictionary mapping classes (str) to their corresponding color (iterable of [R, G, B])
    :return: a dictionary mapping classes (str) to their corresponding
        bboxes (a dictionary with the keys 'xmin', 'xmax', 'ymin', 'ymax'). If a class does not appear in the image,
        then it will not appear in the keys of the returned dictionary.
    """
    img = np.array(Image.open(mask_path))[..., :3]  # remove alpha channel
    bboxes = {}
    for class_name, color in label_map.items():
        color = np.array(list(color))
        class_mask = np.all(img == color, axis=-1)
        ys, xs = class_mask.nonzero()
        if len(ys) > 0:
            bboxes[class_name] = {
                'ymin': int(np.min(ys)),
                'ymax': int(np.max(ys)),
                'xmin': int(np.min(xs)),
                'xmax': int(np.max(xs)),
            }
    return bboxes


def get_centroids_from_mask(mask_path, label_map):
    """
    Gets bounding boxes from instance masks.
    :param mask_path: path to mask image (str)
    :param label_map: dictionary mapping classes (str) to their corresponding color (iterable of [R, G, B])
    :return: a dictionary mapping classes (str) to their corresponding
        centroids (y, x). If a class does not appear in the image,
        then it will not appear in the keys of the returned dictionary.
    """
    img = np.array(Image.open(mask_path))[..., :3]  # remove alpha channel
    centroids = {}
    for class_name, color in label_map.items():
        color = np.array(list(color))
        class_mask = np.all(img == color, axis=-1)
        ys, xs = class_mask.nonzero()
        if len(ys) > 0:
            centroids[class_name] = (int(np.mean(ys)), int(np.mean(xs)))
    return centroids
