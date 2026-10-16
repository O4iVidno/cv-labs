import os
from pathlib import Path

import csv

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gauss import convolution_numpy

os.chdir(Path(__file__).resolve().parent.parent)

matplotlib.use('Agg')

OUT = 'results/plots'
# фиксированный порядок цветов, цвет закреплен за реализацией
COLORS = {'L1': '#2a78d6', 'L2': '#eb6834', 'N1': '#1baf7a', 'CV': '#eda100'}
NAMES = {
    'L1': 'L1 циклы, 2D ядро',
    'L2': 'L2 циклы, разделимое',
    'N1': 'N1 numpy, разделимое',
    'CV': 'CV cv2.GaussianBlur',
}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.size': 11, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.dpi': 110, 'savefig.dpi': 150,
})


def sigma_for(k):
    # формула OpenCV для sigma по размеру ядра
    return 0.3 * ((k - 1) * 0.5 - 1) + 0.8


def load():
    with open('results/bench.csv') as f:
        return list(csv.DictReader(f))


def fmt_time(t):
    return f'{t:.2f} с' if t >= 1 else f'{t * 1000:.2f} мс'


def line_chart(rows, series, xkey, xlabel, title, fname, logx, legend_at=(0.0, 1.0)):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for impl in COLORS:
        pts = sorted((int(r[xkey]), float(r['seconds'])) for r in rows
                     if r['series'] == series and r['impl'] == impl)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=COLORS[impl], lw=2, marker='o', ms=7,
                markeredgecolor='white', markeredgewidth=1.5, label=NAMES[impl])
        # подпись у последней точки, текст чернилами, не цветом серии
        ax.annotate(f'{impl}  {fmt_time(ys[-1])}', (xs[-1], ys[-1]),
                    xytext=(8, 0), textcoords='offset points', va='center',
                    color=INK, fontsize=10)
    ax.set_yscale('log')
    if logx:
        ax.set_xscale('log', base=2)
    xs_all = sorted({int(r[xkey]) for r in rows if r['series'] == series})
    ax.set_xticks(xs_all)
    ax.set_xticklabels([str(x) for x in xs_all])
    ax.minorticks_off() if logx else None
    ax.set_xlabel(xlabel)
    ax.set_ylabel('время, с (логарифмическая шкала)')
    ax.set_title(title, loc='left', color=INK, fontsize=13)
    ax.grid(True, which='major', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=legend_at, fontsize=10)
    ax.set_xlim(right=xs_all[-1] * (1.9 if logx else 1.22))
    fig.tight_layout()
    fig.savefig(f'{OUT}/{fname}')
    plt.close(fig)


def gallery(images, titles, fname, suptitle, cols=None):
    cols = cols or len(images)
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.4 * rows))
    for ax, img, t in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax.set_title(t, fontsize=10, color=INK)
        ax.axis('off')
    for ax in np.atleast_1d(axes).ravel()[len(images):]:
        ax.axis('off')
    fig.suptitle(suptitle, x=0.01, ha='left', color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{OUT}/{fname}')
    plt.close(fig)


rows = load()
line_chart(rows, 'size', 'size', 'сторона квадратной картинки, пикселей',
           'Время от размера картинки, ядро 7 на 7', 'time_vs_size.png', True)
line_chart(rows, 'kernel', 'kernel', 'размер ядра k',
           'Время от размера ядра, картинка 256 на 256', 'time_vs_kernel.png', False, legend_at=(0.0, 0.62))

cam = cv2.imread('data/camera_512.png', cv2.IMREAD_GRAYSCALE)

# сила размытия от размера ядра, sigma по формуле
ks = [3, 7, 15, 31]
gallery([cam] + [convolution_numpy(cam, k, sigma_for(k)) for k in ks],
        ['оригинал'] + [f'k={k}, sigma={sigma_for(k):.1f}' for k in ks],
        'blur_by_kernel.png', 'Размытие при росте ядра, sigma по формуле OpenCV')

# влияние sigma при одном ядре
k = 21
sig = [0.5, 1.5, 3.5, 10.0]
gallery([convolution_numpy(cam, k, s) for s in sig],
        [f'k={k}, sigma={s}' for s in sig],
        'blur_by_sigma.png', 'Влияние sigma при фиксированном ядре 21 на 21')

# темная рамка от дополнения нулями, угол крупно
crop = lambda a: cv2.resize(a[:64, :64], (256, 256), interpolation=cv2.INTER_NEAREST)
gallery([crop(cam)] + [crop(convolution_numpy(cam, k, sigma_for(k))) for k in (15, 31)],
        ['оригинал, угол 64 на 64', 'k=15, рамка 7 пикселей', 'k=31, рамка 15 пикселей'],
        'border_effect.png', 'Рамка от дополнения нулями, левый верхний угол, увеличено в 4 раза')

# шум двух видов до и после фильтра
g = cv2.imread('data/camera_gauss.png', cv2.IMREAD_GRAYSCALE)
sp = cv2.imread('data/camera_saltpepper.png', cv2.IMREAD_GRAYSCALE)
k = 5
crop2 = lambda a: a[100:300, 150:350]
gallery([crop2(g), crop2(convolution_numpy(g, k, sigma_for(k))),
         crop2(sp), crop2(convolution_numpy(sp, k, sigma_for(k)))],
        ['гауссов шум', 'после фильтра, k=5', 'соль и перец', 'после фильтра, k=5'],
        'noise.png', 'Подавление шума, фрагмент 200 на 200', cols=4)

# импульс, видно само ядро
imp = cv2.imread('data/impulse_8x8.png', cv2.IMREAD_GRAYSCALE)
fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
for ax, a, t in zip(axes, [imp, convolution_numpy(imp, 3, 0.849322)],
                    ['импульс 8 на 8', 'после фильтра, k=3, яркость растянута до 64']):
    ax.imshow(a, cmap='gray', vmin=0, vmax=255 if a is imp else 64, interpolation='nearest')
    for (y, x), v in np.ndenumerate(a):
        if v:
            ax.text(x, y, str(v), ha='center', va='center', fontsize=9,
                    color='black' if v > (128 if a is imp else 32) else 'white')
    ax.set_title(t, fontsize=10, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Импульсный отклик, пятно повторяет ядро', x=0.01, ha='left', color=INK, fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUT}/impulse.png')
plt.close(fig)

print('готово:', OUT)
