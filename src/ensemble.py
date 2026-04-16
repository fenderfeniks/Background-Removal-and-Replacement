import numpy as np

def ensemble_masks(mask1, mask2, w1=0.85, w2=0.15):
    m1 = mask1.astype(np.float32) / 255.0
    m2 = mask2.astype(np.float32) / 255.0

    combined = w1 * m1 + w2 * m2
    combined = np.clip(combined, 0, 1)

    return (combined * 255).astype(np.uint8)