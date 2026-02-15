from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.services.style_registry import AQEPreset


def decode_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.array(image)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Unable to decode image input.") from exc

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def resize_longest_side(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image_bgr

    scale = max_side / float(longest)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def lab_percentile_stretch(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)

    low, high = np.percentile(l_channel, (1, 99))
    if high <= low:
        return image_bgr

    stretched = np.clip((l_channel - low) * (255.0 / (high - low)), 0, 255)
    lab[:, :, 0] = stretched.astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def bilateral_denoise(image_bgr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return image_bgr

    diameter = int(5 + strength * 10)
    sigma = float(25 + strength * 70)
    return cv2.bilateralFilter(image_bgr, d=max(3, diameter), sigmaColor=sigma, sigmaSpace=sigma)


def preprocess_baseline(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    return resize_longest_side(image_bgr, max_side=max_side)


def preprocess_improved(image_bgr: np.ndarray, preset: AQEPreset) -> np.ndarray:
    resized = resize_longest_side(image_bgr, max_side=preset.resize_max)
    luminance_balanced = lab_percentile_stretch(resized)
    denoised = bilateral_denoise(luminance_balanced, preset.denoise_strength)
    return denoised
