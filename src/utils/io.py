from PIL import Image
import numpy as np
import os

# Загрузка изображения
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    image = Image.open(image_path).convert('RGB')

    return np.array(image)

# Сохранение изображения
def save_image(image, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    Image.fromarray(image).save(path)



# Загрузка маски
def load_mask(mask_path):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Файл не найден: {mask_path}")

    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask, dtype=np.uint8)

    return mask