import numpy as np


def binarize_mask(mask, threshold=127):
    return (mask > threshold).astype(np.uint8)


def iou_score(pred_mask, true_mask, threshold=127):
    pred = binarize_mask(pred_mask, threshold)
    true = binarize_mask(true_mask, threshold)

    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()

    if union == 0:
        return 1.0

    return intersection / union

def dice_score(pred_mask, true_mask, threshold=127):
    pred = binarize_mask(pred_mask, threshold)
    true = binarize_mask(true_mask, threshold)

    intersection = np.logical_and(pred, true).sum()
    pred_sum = pred.sum()
    true_sum = true.sum()

    if pred_sum + true_sum == 0:
        return 1.0

    return 2 * intersection / (pred_sum + true_sum)

def mad_score (pred_mask, true_mask):
    pred = pred_mask.astype(np.float32) / 255.0
    true = true_mask.astype(np.float32) / 255.0
    return np.mean(np.abs(pred - true))