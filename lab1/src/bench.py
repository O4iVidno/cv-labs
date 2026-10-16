import os
from pathlib import Path

import csv
import statistics
import time

import cv2
import numpy as np

from gauss import convolution, convolution_separable, convolution_numpy

os.chdir(Path(__file__).resolve().parent.parent)


def sigma_for(k):
    # формула OpenCV для sigma по размеру ядра
    return 0.3 * ((k - 1) * 0.5 - 1) + 0.8


def run_cv(img, k, s):
    return cv2.GaussianBlur(img, (k, k), s, borderType=cv2.BORDER_CONSTANT)


IMPLS = {
    'L1': (convolution, 3, False),
    'L2': (convolution_separable, 3, False),
    'N1': (convolution_numpy, 21, True),
    'CV': (run_cv, 21, True),
}


def measure(name, img, k):
    func, repeats, warmup = IMPLS[name]
    s = sigma_for(k)
    # холостой вызов, в результат не идет
    if warmup:
        func(img, k, s)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(img, k, s)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


rows = []

# серия 1, время от размера картинки, ядро 7
for size in (64, 128, 256, 512, 1024):
    img = cv2.imread(f'data/bench_{size}.png', cv2.IMREAD_GRAYSCALE)
    for name in IMPLS:
        # L1 на 1024 слишком долгий
        if name == 'L1' and size > 512:
            continue
        t = measure(name, img, 7)
        rows.append(('size', name, size, 7, t))
        print(f'размер {size:5}  k=7   {name}  {t:10.5f} с', flush=True)

# серия 2, время от размера ядра, картинка 256
img = cv2.imread('data/bench_256.png', cv2.IMREAD_GRAYSCALE)
for k in (3, 5, 7, 9, 11, 15, 21, 31):
    for name in IMPLS:
        t = measure(name, img, k)
        rows.append(('kernel', name, 256, k, t))
        print(f'размер   256  k={k:<3} {name}  {t:10.5f} с', flush=True)

with open('results/bench.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['series', 'impl', 'size', 'kernel', 'seconds'])
    w.writerows(rows)
