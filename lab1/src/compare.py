import os
from pathlib import Path

import cv2
import numpy as np

os.chdir(Path(__file__).resolve().parent.parent)


def compare(name, first, second):
    # обе картинки в int, чтобы разность не завернулась как в uint8
    a = cv2.imread(f'results/{first}/{name}.png', cv2.IMREAD_GRAYSCALE).astype(int)
    b = cv2.imread(f'results/{second}/{name}.png', cv2.IMREAD_GRAYSCALE).astype(int)

    # карта разности по модулю, все метрики считаются от нее
    diff = np.abs(a - b)

    print(f'{first} против {second}, {name}')
    print('  совпадают полностью:', np.array_equal(a, b))
    print('  максимальное расхождение:', diff.max())
    print('  среднее расхождение:', round(diff.mean(), 5))
    # True у каждого несовпавшего пикселя, mean дает долю
    print('  отличается пикселей, %:', round((diff > 0).mean() * 100, 2))


names = ['impulse_k3', 'gradient_k3', 'camera_k3', 'camera_k15']

for name in names:
    compare(name, 'L1', 'CV')

for name in names:
    compare(name, 'L2', 'L1')

for name in names:
    compare(name, 'L2', 'CV')

for name in names:
    compare(name, 'N1', 'L1')

for name in names:
    compare(name, 'N1', 'CV')
