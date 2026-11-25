import numpy as np

from upscale import upscale_mask


FIELDS = ('psnr', 'lpips', 'clip_iqa', 'psnr_mask', 'lpips_crop')


def load_metrics():
    import lpips
    from torchmetrics.multimodal import CLIPImageQualityAssessment as CLIP_IQA

    # Метрики считаются на CPU, чтобы оставить память видеокарты апскейлеру
    lpips_model = lpips.LPIPS(net='alex', verbose=False)
    lpips_model.eval()

    clip_iqa_model = CLIP_IQA()
    clip_iqa_model.eval()

    return lpips_model, clip_iqa_model


def psnr(difference):
    # Квадрат не дает положительным и отрицательным ошибкам погасить друг друга
    squared_error = difference ** 2
    mean_squared_error = float(np.mean(squared_error))

    # Бесконечный PSNR означает точное совпадение пикселей
    if mean_squared_error == 0:
        score = float('inf')
    else:
        # Сравниваем максимальную и среднюю квадратичную ошибку в децибелах
        peak_squared = 255 ** 2
        score = float(10 * np.log10(peak_squared / mean_squared_error))

    return score


def image_tensor(image):
    import torch

    pixels = np.array(image, dtype=np.float32)
    pixels = pixels / 255
    tensor = torch.from_numpy(pixels)

    # В фото каналы идут последними, а модели PyTorch ждут их перед высотой и шириной
    tensor = tensor.permute(2, 0, 1)

    # Добавляем ось батча из одной картинки и получаем 1 × C × H × W
    tensor = tensor.unsqueeze(0)

    return tensor


def measure(result, reference, mask, lpips_model, clip_iqa_model):
    import torch

    # Переходим к float, чтобы разность пикселей могла быть отрицательной
    result_pixels = np.asarray(result, dtype=np.float32)
    reference_pixels = np.asarray(reference, dtype=np.float32)
    difference = result_pixels - reference_pixels

    mask_image = upscale_mask(mask, result.size)
    selected = np.asarray(mask_image, dtype=bool)

    result_tensor = image_tensor(result)
    reference_tensor = image_tensor(reference)

    # LPIPS принимает значения от -1 до 1, а CLIP-IQA – от 0 до 1
    result_lpips = result_tensor * 2 - 1
    reference_lpips = reference_tensor * 2 - 1

    psnr_full = psnr(difference)
    # При пустой маске оценка выбранной области остается пустой
    psnr_mask = None
    has_selected_pixels = selected.any()
    if has_selected_pixels:
        selected_difference = difference[selected]
        psnr_mask = psnr(selected_difference)

    lpips_crop = None
    with torch.inference_mode():
        lpips_full = lpips_model(result_lpips, reference_lpips).item()
        clip_iqa_score = clip_iqa_model(result_tensor).item()
        # Сбрасываем состояние CLIP-IQA перед следующей картинкой
        clip_iqa_model.reset()

        # Если маска покрывает весь кадр, используем уже посчитанный LPIPS
        if selected.all():
            lpips_crop = lpips_full
        elif has_selected_pixels:
            # LPIPS сравнивает прямоугольник вокруг выбранных пикселей
            rows, columns = np.where(selected)
            left = int(columns.min())
            right = int(columns.max()) + 1
            top = int(rows.min())
            bottom = int(rows.max()) + 1

            crop_width = right - left
            crop_height = bottom - top
            # Слишком маленькая вырезка не пройдет через слои AlexNet
            if crop_width >= 32 and crop_height >= 32:
                result_crop = result_lpips[:, :, top:bottom, left:right]
                reference_crop = reference_lpips[:, :, top:bottom, left:right]
                lpips_crop = lpips_model(result_crop, reference_crop).item()

    scores = {
        'psnr': psnr_full,
        'lpips': lpips_full,
        'clip_iqa': clip_iqa_score,
        'psnr_mask': psnr_mask,
        'lpips_crop': lpips_crop,
    }

    return scores
