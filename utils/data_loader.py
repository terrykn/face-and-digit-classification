import numpy as np
from typing import Tuple

def stream_load_data(file_path: str, label_path: str, image_size: Tuple[int, int]):
    with open(label_path, 'r') as f:
        labels = np.array([int(line.strip()) for line in f], dtype=np.uint8)
    num_images = len(labels)
    images = np.zeros((num_images, image_size[0] * image_size[1]), dtype=np.uint8)
    with open(file_path, 'r') as f:
        for i in range(num_images):
            img_lines = [next(f) for _ in range(image_size[0])]
            img = ''.join(img_lines).replace(' ', '0').replace('#', '1').replace('+', '1')
            images[i] = np.array([int(pix) for pix in img.strip().replace('\n', '')], dtype=np.uint8)
    return images.astype(np.float32), labels