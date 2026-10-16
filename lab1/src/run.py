import os
from pathlib import Path

import cv2

from gauss import convolution, convolution_separable, convolution_numpy

os.chdir(Path(__file__).resolve().parent.parent)

img_impulse_8x8 = cv2.imread('data/impulse_8x8.png', cv2.IMREAD_GRAYSCALE)
img_gradient_128 = cv2.imread('data/gradient_128.png', cv2.IMREAD_GRAYSCALE)
img_camera_512 = cv2.imread('data/camera_512.png', cv2.IMREAD_GRAYSCALE)

# L1, своя свертка
l1_impulse_k3 = convolution(img_impulse_8x8, 3, 0.849322)
cv2.imwrite('results/L1/impulse_k3.png', l1_impulse_k3)

l1_gradient_k3 = convolution(img_gradient_128, 3, 0.8)
cv2.imwrite('results/L1/gradient_k3.png', l1_gradient_k3)

l1_camera_k3 = convolution(img_camera_512, 3, 0.8)
cv2.imwrite('results/L1/camera_k3.png', l1_camera_k3)

l1_camera_k15 = convolution(img_camera_512, 15, 2.6)
cv2.imwrite('results/L1/camera_k15.png', l1_camera_k15)


# L2, своя свертка с разделимым ядром
l2_impulse_k3 = convolution_separable(img_impulse_8x8, 3, 0.849322)
cv2.imwrite('results/L2/impulse_k3.png', l2_impulse_k3)

l2_gradient_k3 = convolution_separable(img_gradient_128, 3, 0.8)
cv2.imwrite('results/L2/gradient_k3.png', l2_gradient_k3)

l2_camera_k3 = convolution_separable(img_camera_512, 3, 0.8)
cv2.imwrite('results/L2/camera_k3.png', l2_camera_k3)

l2_camera_k15 = convolution_separable(img_camera_512, 15, 2.6)
cv2.imwrite('results/L2/camera_k15.png', l2_camera_k15)


# N1, своя свертка на numpy, разделимое ядро
n1_impulse_k3 = convolution_numpy(img_impulse_8x8, 3, 0.849322)
cv2.imwrite('results/N1/impulse_k3.png', n1_impulse_k3)

n1_gradient_k3 = convolution_numpy(img_gradient_128, 3, 0.8)
cv2.imwrite('results/N1/gradient_k3.png', n1_gradient_k3)

n1_camera_k3 = convolution_numpy(img_camera_512, 3, 0.8)
cv2.imwrite('results/N1/camera_k3.png', n1_camera_k3)

n1_camera_k15 = convolution_numpy(img_camera_512, 15, 2.6)
cv2.imwrite('results/N1/camera_k15.png', n1_camera_k15)


# CV, библиотечная свертка с теми же параметрами и краями нулями
cv_impulse_k3 = cv2.GaussianBlur(img_impulse_8x8, (3, 3), 0.849322, borderType=cv2.BORDER_CONSTANT)
cv2.imwrite('results/CV/impulse_k3.png', cv_impulse_k3)

cv_gradient_k3 = cv2.GaussianBlur(img_gradient_128, (3, 3), 0.8, borderType=cv2.BORDER_CONSTANT)
cv2.imwrite('results/CV/gradient_k3.png', cv_gradient_k3)

cv_camera_k3 = cv2.GaussianBlur(img_camera_512, (3, 3), 0.8, borderType=cv2.BORDER_CONSTANT)
cv2.imwrite('results/CV/camera_k3.png', cv_camera_k3)

cv_camera_k15 = cv2.GaussianBlur(img_camera_512, (15, 15), 2.6, borderType=cv2.BORDER_CONSTANT)
cv2.imwrite('results/CV/camera_k15.png', cv_camera_k15)

