import numpy as np
from PIL import Image


MODEL_ID = 'stabilityai/stable-diffusion-x4-upscaler'
SCALE = 4
# Уровень шума, который модель добавляет к маленькому входному фото
NOISE_LEVEL = 20


def load_model(device):
    import torch
    from diffusers import DDIMScheduler, StableDiffusionUpscalePipeline

    device = torch.device(device)
    dtype = torch.float32
    variant = None
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA недоступна. Проверьте видеокарту и драйвер через nvidia-smi.')
        # Половинная точность экономит память видеокарты
        dtype = torch.float16
        variant = 'fp16'

    # Загружаем уже обученную модель с готовыми весами
    model = StableDiffusionUpscalePipeline.from_pretrained(
        MODEL_ID,
        use_safetensors=True,
        variant=variant,
        torch_dtype=dtype,
    )
    model = model.to(device)
    if not isinstance(model.scheduler, DDIMScheduler):
        raise RuntimeError('Подстановка фона рассчитана на шаги DDIM.')

    # VAE считает в float32, чтобы избежать переполнения
    model.vae.to(dtype=torch.float32)
    model.set_progress_bar_config(disable=True)
    return model


def bicubic(image):
    width, height = image.size
    output_size = (width * SCALE, height * SCALE)
    # Бикубическая копия служит фоном во всех режимах
    background = image.resize(output_size, Image.Resampling.BICUBIC)
    return background


def upscale_mask(mask, size):
    # Переводим значения 0 и 1 в черный и белый для Pillow
    mask_pixels = np.asarray(mask, dtype=np.uint8) * 255
    mask_image = Image.fromarray(mask_pixels)
    # Ближайший сосед сохраняет жесткую границу маски
    resized_mask = mask_image.resize(size, Image.Resampling.NEAREST)
    return resized_mask


def compose(result, background, mask):
    if result.size != background.size:
        raise ValueError('Размер результата отличается от размера фона.')

    # Пиксели снаружи маски берутся из исходной бикубической копии
    mask_image = upscale_mask(mask, result.size)
    combined = Image.composite(result, background, mask_image)
    return combined


def run_model(model, image, *, seed, steps, after_step=None):
    import torch

    # Один seed дает одинаковый стартовый шум для сравнения масок
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)

    # Запускаем готовую модель без расчета градиентов
    with torch.inference_mode():
        output = model(
            prompt='',
            image=image,
            generator=generator,
            # Число шагов DDIM, на каждом меняется латент картинки
            num_inference_steps=steps,
            guidance_scale=1.0,
            noise_level=NOISE_LEVEL,
            output_type='pil',
            callback=after_step,
            callback_steps=1,
        )

    result = output.images[0]
    return result


def upscale_masked(model, image, mask, *, seed, steps):
    import torch

    image = image.convert('RGB')
    mask = np.asarray(mask)
    if mask.shape != (image.height, image.width):
        raise ValueError('Маска должна совпадать с размером фото.')
    if not np.isin(mask, (0, 1)).all():
        raise ValueError('Маска должна состоять из 0 и 1.')
    if image.width % 8 or image.height % 8:
        raise ValueError('Ширина и высота входного изображения должны быть кратны 8.')

    background = bicubic(image)
    device = model.unet.device
    dtype = model.unet.dtype

    # Подготавливаем бикубический фон для энкодера VAE
    pixels = model.image_processor.preprocess(background)
    pixels = pixels.to(device=device, dtype=model.vae.dtype)

    with torch.inference_mode():
        # Среднее распределения VAE дает повторяемый латент фона
        encoded = model.vae.encode(pixels)
        background_latent = encoded.latent_dist.mode()
        background_latent = background_latent * model.vae.config.scaling_factor
        background_latent = background_latent.to(dtype=dtype)

    if background_latent.shape[-2:] != mask.shape:
        raise RuntimeError('Размер латента фона отличается от размера маски.')

    # Добавляем оси батча и каналов, чтобы одна маска работала для всех каналов
    mask_values = mask.astype(bool)
    latent_mask = torch.as_tensor(mask_values, device=device)
    latent_mask = latent_mask.unsqueeze(0)
    latent_mask = latent_mask.unsqueeze(0)

    # Отдельный генератор сохраняет штатный начальный шум SD x4
    background_seed = seed ^ 0x9E3779B9
    generator = torch.Generator(device='cpu')
    generator.manual_seed(background_seed)
    background_noise = torch.randn(background_latent.shape, generator=generator)
    background_noise = background_noise.to(device=device, dtype=dtype)

    scheduler = model.scheduler
    # Отдельный шедулер добавляет шум к фону по тому же расписанию DDIM
    scheduler_type = type(scheduler)
    background_scheduler = scheduler_type.from_config(scheduler.config)
    masked_steps = []

    def replace_background(step, _timestep, latents):
        next_step = step + 1
        timesteps = scheduler.timesteps

        # На последнем шаге подставляется чистый фон
        replacement = background_latent
        # Уровень шума фона соответствует входу следующего шага
        if next_step < len(timesteps):
            next_timestep = timesteps[next_step:next_step + 1]
            replacement = background_scheduler.add_noise(
                background_latent, background_noise, next_timestep
            )

        # Под единицами оставляем результат модели, под нулями подставляем фон
        combined_latents = torch.where(latent_mask, latents, replacement)
        # Перезаписываем тензор, который Diffusers передаст на следующий шаг
        latents.copy_(combined_latents)
        masked_steps.append(step)

    result = run_model(model, image, seed=seed, steps=steps, after_step=replace_background)
    # Проверяем, что подстановка сработала после каждого шага DDIM
    number_of_replacements = len(masked_steps)
    if number_of_replacements != steps:
        raise RuntimeError(f'Фон подставлен {number_of_replacements} раз, ожидалось {steps}.')

    # Повторяем сборку в RGB, чтобы VAE не изменил пиксели фона
    combined = compose(result, background, mask)
    return combined
