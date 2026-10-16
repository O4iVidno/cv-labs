import numpy as np
import cv2


def calculate_gaussian_kernel(kernel_size, sigma):
    if kernel_size % 2 == 0:
        raise ValueError ("The kernel size must be odd.")
    arr = np.zeros((kernel_size, kernel_size))
    r = (kernel_size - 1) / 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            arr[i, j] = np.exp(-((i-r)**2+(j-r)**2)/(2*sigma**2))

    sum_matrix = np.sum(arr)
    arr /= sum_matrix
    return arr


def convolution(img, kernel_size, sigma):
    r = (kernel_size - 1) // 2
    img_zero = np.zeros_like(img, dtype=float)
    pad = np.pad(img, pad_width=r)
    kernel = calculate_gaussian_kernel(kernel_size,sigma)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            total = 0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    total += kernel[ki][kj] * pad[i + ki, j + kj]
            img_zero[i, j] = total
    img_lim = np.clip(img_zero, 0, 255)
    img_round = np.round(img_lim)
    return np.uint8(img_round)


def calculate_gaussian_kernel_1d(kernel_size, sigma):
    if kernel_size % 2 == 0:
        raise ValueError ("The kernel size must be odd.")

    r = (kernel_size - 1) / 2
    arr = np.zeros(kernel_size)
    for i in range(kernel_size):
        arr[i] = np.exp(-((i - r)**2 / (2 * sigma**2)))

    sum_matrix = np.sum(arr)
    arr /= sum_matrix
    return arr

def convolution_separable(img, kernel_size, sigma):
    r = (kernel_size - 1) // 2
    tmp = np.zeros_like(img, dtype=float)
    out = tmp.copy()
    pad_v = np.pad(img, pad_width=((0,0),(r,r)))
    kernel = calculate_gaussian_kernel_1d(kernel_size, sigma)
    W = img.shape[0]
    H = img.shape[1]

    for i in range(W):
        for j in range(H):
            total = 0
            for k in range(kernel_size):
                total += kernel[k] * pad_v[i, j + k]
            tmp[i, j] = total

    pad_h = np.pad(tmp, pad_width=((r,r),(0,0)))
    for i in range(W):
        for j in range(H):
            total = 0
            for k in range(kernel_size):
                total += kernel[k] * pad_h[i + k, j]
            out[i, j] = total

    img_lim = np.clip(out, 0, 255)
    img_round = np.round(img_lim)
    return np.uint8(img_round)


def convolution_numpy(img, kernel_size, sigma):
    r = (kernel_size - 1) // 2
    img_zero = np.zeros_like(img, dtype=float)
    tmp = img_zero.copy()
    out = tmp.copy()
    pad_v = np.pad(img, pad_width=((0, 0), (r, r)))
    kernel = calculate_gaussian_kernel_1d(kernel_size, sigma)
    H = img.shape[0]
    W = img.shape[1]

    for k in range(len(kernel)):
        tmp += kernel[k] * pad_v[:, k : k + W]

    pad_h = np.pad(tmp, pad_width=((r,r),(0,0)))
    for k in range(len(kernel)):
        out += kernel[k] * pad_h[k : k + H, :]

    img_lim = np.clip(out, 0, 255)
    img_round = np.round(img_lim)
    return np.uint8(img_round)
