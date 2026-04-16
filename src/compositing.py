import numpy as np
from PIL import Image

# Замена фона на основании полученной маски
def replace_background(image_front, image_back, mask):
    if image_front.shape[:2] != mask.shape[:2] or image_front.shape[:2] != image_back.shape[:2]:
        raise ValueError("Размеры не совпадают")

    alpha = mask.astype(np.float32) / 255.0
    alpha = alpha[..., None]

    image_front = image_front.astype(np.float32)
    image_back = image_back.astype(np.float32)

    result = image_front * alpha + image_back * (1.0 - alpha)

    return result.astype(np.uint8)

# подготовка фона
def prepare_background_image(background_image, target_shape):
    target_h, target_w = target_shape[:2]
    bg_pil = Image.fromarray(background_image)
    bg_pil = bg_pil.resize((target_w, target_h))
    return np.array(bg_pil, dtype=np.uint8)

# формирование однотонного заднего фона
def create_solid_background(image_shape, color=(255, 255, 255)):
    h, w = image_shape[:2]
    background = np.ones((h, w, 3), dtype=np.uint8)
    background[:, :] = color
    return background

# Применение маски(удаление фона)
def apply_mask(image, mask):
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Размеры не совпадают: image={image.shape[:2]}, mask={mask.shape[:2]}")

    return np.dstack((image, mask))