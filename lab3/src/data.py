from io import BytesIO
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile, is_zipfile

import numpy as np
from PIL import Image, ImageFilter

from upscale import SCALE


DIV2K_URL = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/'
# HR – оригиналы, LR – уменьшенные версии тех же фото
ARCHIVES = (
    'DIV2K_valid_HR.zip',
    'DIV2K_valid_LR_bicubic_X4.zip',
    'DIV2K_valid_LR_unknown_X4.zip',
)

# Сторона квадратной вырезки на маленьком фото
PATCH_SIZE = 128

# Номер фото и границы маски на вырезке LR (слева, сверху, справа, снизу)
SCENES = {
    '0801': (24, 22, 96, 100),
    '0802': (34, 24, 108, 108),
    '0803': (18, 35, 100, 111),
}


# Один вход LR, его эталон HR и три маски
class Case:
    def __init__(self, scene, degradation, image, reference, masks):
        self.scene = scene
        self.degradation = degradation
        self.image = image
        self.reference = reference
        self.masks = masks


def make_masks(size, box):
    w, h = size
    left, top, right, bottom = box
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise ValueError(f'Прямоугольник {box} выходит за границы изображения {size}.')
    mixed = np.zeros((h, w), dtype=np.uint8)
    # В NumPy сначала строки (y), потом столбцы (x)
    mixed[top:bottom, left:right] = 1
    zero = np.zeros_like(mixed)
    one = np.ones_like(mixed)
    masks = {'zero': zero, 'one': one, 'mixed': mixed}
    return masks


def download_archive(data_dir, name):
    target = data_dir / name
    if is_zipfile(target):
        return target

    data_dir.mkdir(parents=True, exist_ok=True)
    url = DIV2K_URL + name

    # Временная папка удаляется даже при обрыве загрузки
    with TemporaryDirectory(dir=data_dir) as temporary_dir:
        temporary_path = Path(temporary_dir) / name
        with urlopen(url, timeout=90) as response:
            with temporary_path.open('wb') as file:
                copyfileobj(response, file, length=1024 * 1024)

        if not is_zipfile(temporary_path):
            raise ValueError(f'Получен некорректный архив {name}.')
        temporary_path.replace(target)

    return target


def read_image(archive, name):
    for entry in archive.namelist():
        # Сравнение по имени файла без папок внутри ZIP
        filename = Path(entry).name
        if filename == name:
            image_bytes = archive.read(entry)
            with Image.open(BytesIO(image_bytes)) as image:
                rgb = image.convert('RGB')
            return rgb

    raise FileNotFoundError(f'В архиве {archive.filename} отсутствует {name}.')


def crop_pair(hr, bicubic, unknown):
    w = min(hr.width // SCALE, bicubic.width, unknown.width)
    h = min(hr.height // SCALE, bicubic.height, unknown.height)
    if min(w, h) < PATCH_SIZE:
        raise ValueError('Изображение слишком маленькое для общей вырезки.')

    # Центральная вырезка в координатах LR
    left = (w - PATCH_SIZE) // 2
    top = (h - PATCH_SIZE) // 2
    right = left + PATCH_SIZE
    bottom = top + PATCH_SIZE
    box_lr = (left, top, right, bottom)
    # В HR координаты той же вырезки в четыре раза больше
    box_hr = (left * SCALE, top * SCALE, right * SCALE, bottom * SCALE)

    reference = hr.crop(box_hr)
    image_bicubic = bicubic.crop(box_lr)
    image_unknown = unknown.crop(box_lr)
    return reference, image_bicubic, image_unknown


def degrade_realesrgan(image, seed):
    rng = np.random.default_rng(seed)
    target_size = (image.width // SCALE, image.height // SCALE)

    # Два цикла размытия, изменения размера, шума и JPEG по схеме Real-ESRGAN
    for low, high in ((0.6, 0.85), (0.45, 0.75)):
        blur_radius = float(rng.uniform(0.6, 1.8))
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))

        scale = rng.uniform(low, high)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        methods = (Image.Resampling.BICUBIC, Image.Resampling.BILINEAR)
        method = methods[rng.integers(2)]
        image = image.resize((width, height), method)

        pixels = np.asarray(image, dtype=np.float32)
        noise_std = rng.uniform(2, 9)
        noise = rng.normal(0, noise_std, size=pixels.shape)
        noisy_pixels = np.rint(pixels + noise)
        # После добавления шума значения ограничиваются диапазоном 0–255
        noisy_pixels = np.clip(noisy_pixels, 0, 255).astype(np.uint8)
        image = Image.fromarray(noisy_pixels)

        # JPEG записывается в память
        buffer = BytesIO()
        jpeg_quality = int(rng.integers(45, 91))
        image.save(buffer, format='JPEG', quality=jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            image = compressed.convert('RGB')

    degraded = image.resize(target_size, Image.Resampling.BICUBIC)
    return degraded


def prepare_cases(data_dir):
    hr_name, bicubic_name, unknown_name = ARCHIVES
    hr_path = download_archive(data_dir, hr_name)
    bicubic_path = download_archive(data_dir, bicubic_name)
    unknown_path = download_archive(data_dir, unknown_name)

    cases = []
    with (
        ZipFile(hr_path) as originals,
        ZipFile(bicubic_path) as bicubic,
        ZipFile(unknown_path) as unknown,
    ):
        for scene, box in SCENES.items():
            hr = read_image(originals, f'{scene}.png')
            lr_bicubic = read_image(bicubic, f'{scene}x4.png')
            lr_unknown = read_image(unknown, f'{scene}x4.png')

            reference, image_bicubic, image_unknown = crop_pair(hr, lr_bicubic, lr_unknown)
            # Номер фото служит seed для синтетической деградации
            image_synthetic = degrade_realesrgan(reference, seed=int(scene))

            # Маски общие для всех трех деградаций сцены
            masks = make_masks(image_bicubic.size, box)

            cases.append(Case(scene, 'bicubic', image_bicubic, reference, masks))
            cases.append(Case(scene, 'unknown', image_unknown, reference, masks))
            cases.append(Case(scene, 'realesrgan', image_synthetic, reference, masks))

    return cases
