import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data import Case, make_masks, prepare_cases
from metrics import FIELDS, load_metrics, measure
from upscale import MODEL_ID, NOISE_LEVEL, SCALE, bicubic, load_model, run_model, upscale_mask, upscale_masked


ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = ('scene', 'degradation', 'mask', 'seconds') + FIELDS


def smoke_case():
    photo_path = ROOT.parent / 'lab1' / 'data' / 'camera_512.png'
    with Image.open(photo_path) as photo:
        rgb = photo.convert('RGB')

    reference = rgb.resize((256, 256), Image.Resampling.LANCZOS)
    image = reference.resize((64, 64), Image.Resampling.BICUBIC)
    masks = make_masks(image.size, (17, 10, 48, 39))
    case = Case('camera', 'bicubic', image, reference, masks)
    return case


def save_comparison(folder, case, outputs):
    size, gap, caption = 256, 8, 25
    width = 3 * size + 4 * gap
    height = 2 * (size + caption) + 3 * gap
    board = Image.new('RGB', (width, height), '#f4f4f4')
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=16)
    mixed_mask = upscale_mask(case.masks['mixed'], case.reference.size)
    tiles = (
        ('LR input', case.image),
        ('HR reference', case.reference),
        ('Mixed mask', mixed_mask),
        ('All zeros / bicubic', outputs['zero']),
        ('All ones / full SR', outputs['one']),
        ('Mixed / masked SR', outputs['mixed']),
    )
    for index, (title, image) in enumerate(tiles):
        column = index % 3
        row = index // 3
        x = gap + column * (size + gap)
        y = gap + row * (size + caption + gap)
        draw.text((x, y), title, fill='#111111', font=font)
        method = Image.Resampling.LANCZOS
        if index in (0, 2):
            method = Image.Resampling.NEAREST
        tile = image.resize((size, size), method)
        board.paste(tile, (x, y + caption))
    board.save(folder / 'comparison.png')


def check_masks(model, case, outputs, folder, *, seed, steps):
    expected = bicubic(case.image)
    baseline = run_model(model, case.image, seed=seed, steps=steps)
    baseline.save(folder / 'stock.png')
    background = np.asarray(expected)
    zero = np.asarray(outputs['zero'])
    one = np.asarray(outputs['one'])
    mixed = np.asarray(outputs['mixed'])
    stock = np.asarray(baseline)

    mixed_mask = upscale_mask(case.masks['mixed'], expected.size)
    outside = np.asarray(mixed_mask) == 0
    zero_matches = np.array_equal(zero, background)
    one_matches = np.array_equal(one, stock)
    background_matches = np.array_equal(mixed[outside], background[outside])

    checks = {
        'zero_equals_bicubic': zero_matches,
        'one_equals_stock': one_matches,
        'mixed_keeps_background': background_matches,
    }
    checks_path = folder / 'checks.json'
    checks_text = json.dumps(checks, indent=2)
    checks_path.write_text(checks_text, encoding='utf-8')
    failed = []
    for name, passed in checks.items():
        if not passed:
            failed.append(name)
    if failed:
        raise RuntimeError(f'Проверки не прошли: {", ".join(failed)}. Изображения сохранены в {folder}.')


def run_experiment(mode, data_dir, out_dir, *, seed, steps, device):
    run_dir = out_dir / f'{mode}-{datetime.now():%Y%m%d-%H%M%S-%f}'
    run_dir.mkdir(parents=True)
    settings = {
        'mode': mode, 'model': MODEL_ID, 'device': device, 'seed': seed, 'steps': steps,
        'scale': SCALE, 'noise_level': NOISE_LEVEL, 'prompt': '', 'guidance_scale': 1.0,
    }
    settings_path = run_dir / 'settings.json'
    settings_text = json.dumps(settings, indent=2)
    settings_path.write_text(settings_text, encoding='utf-8')

    print(f'Результаты: {run_dir}', flush=True)
    model = load_model(device)
    if mode == 'smoke':
        cases = [smoke_case()]
    else:
        metrics_path = run_dir / 'metrics.csv'
        with metrics_path.open('w', encoding='utf-8', newline='') as table:
            writer = csv.DictWriter(table, fieldnames=CSV_FIELDS)
            writer.writeheader()
        cases = prepare_cases(data_dir)

    lpips_model = None
    clip_iqa_model = None

    for case in cases:
        folder = run_dir / case.scene / case.degradation
        folder.mkdir(parents=True)
        case.image.save(folder / 'input.png')
        case.reference.save(folder / 'reference.png')

        outputs = {}
        for name, mask in case.masks.items():
            mask_image = Image.fromarray(mask * 255)
            mask_image.save(folder / f'mask_{name}.png')

            start = perf_counter()
            result = upscale_masked(model, case.image, mask, seed=seed, steps=steps)
            elapsed = round(perf_counter() - start, 3)
            result.save(folder / f'{name}.png')
            outputs[name] = result

            if mode == 'full':
                # Модели метрик загружаются один раз после сохранения первого результата
                if lpips_model is None:
                    lpips_model, clip_iqa_model = load_metrics()

                scores = measure(result, case.reference, mask, lpips_model, clip_iqa_model)
                row = {
                    'scene': case.scene,
                    'degradation': case.degradation,
                    'mask': name,
                    'seconds': elapsed,
                }
                row.update(scores)
                with metrics_path.open('a', encoding='utf-8', newline='') as table:
                    writer = csv.DictWriter(table, fieldnames=CSV_FIELDS)
                    writer.writerow(row)

        save_comparison(folder, case, outputs)
        if mode == 'smoke':
            check_masks(model, case, outputs, folder, seed=seed, steps=steps)

    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('smoke', 'full'))
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'div2k')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    steps = args.steps
    if steps is None:
        if args.mode == 'smoke':
            steps = 4
        else:
            steps = 25
    if not 1 <= steps <= 999:
        parser.error('--steps должен быть от 1 до 999')
    if not 0 <= args.seed < 2 ** 32:
        parser.error('--seed должен быть от 0 до 4294967295')
    run_experiment(args.mode, args.data_dir, args.out_dir, seed=args.seed, steps=steps, device=args.device)


if __name__ == '__main__':
    main()
