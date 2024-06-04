import numpy as np
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def compute_dice(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    dice = (2. * np.sum(intersection)) / (np.sum(mask1) + np.sum(mask2))
    return dice