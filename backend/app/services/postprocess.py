from __future__ import annotations

import cv2
import numpy as np

from app.services.style_registry import AQEPreset


def _resize_to_match(image_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
    ref_h, ref_w = reference_bgr.shape[:2]
    img_h, img_w = image_bgr.shape[:2]
    if (img_h, img_w) == (ref_h, ref_w):
        return image_bgr
    return cv2.resize(image_bgr, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)


def reinforce_edges(stylized_bgr: np.ndarray, source_bgr: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        return _resize_to_match(stylized_bgr, source_bgr)

    stylized_bgr = _resize_to_match(stylized_bgr, source_bgr)

    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=90, threshold2=180).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (0, 0), sigmaX=0.9)

    output = stylized_bgr.astype(np.float32)
    output *= 1.0 - (weight * edges[:, :, None])
    return np.clip(output, 0, 255).astype(np.uint8)


def color_quantize(image_bgr: np.ndarray, k: int) -> np.ndarray:
    k = max(4, int(k))
    pixels = image_bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        attempts=3,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    quantized = centers[labels.flatten()].reshape(image_bgr.shape)
    return np.clip(quantized, 0, 255).astype(np.uint8)


def harmonize_contrast_saturation(
    image_bgr: np.ndarray,
    contrast_gain: float,
    saturation_gain: float,
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_gain, 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    lab = cv2.cvtColor(saturated, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * contrast_gain, 0, 255)
    contrasted = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return contrasted


def unsharp_mask(image_bgr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return image_bgr

    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def postprocess_improved(
    stylized_bgr: np.ndarray,
    source_bgr: np.ndarray,
    preset: AQEPreset,
) -> np.ndarray:
    edged = reinforce_edges(stylized_bgr, source_bgr, preset.edge_weight)
    quantized = color_quantize(edged, preset.color_quant_k)
    harmonized = harmonize_contrast_saturation(
        quantized,
        contrast_gain=preset.contrast_gain,
        saturation_gain=preset.saturation_gain,
    )
    sharpened = unsharp_mask(harmonized, preset.sharpen_amount)
    return sharpened


def to_lite_preset(preset: AQEPreset) -> AQEPreset:
    # Conservative AQE configuration focused on reducing artifacts and latency.
    return AQEPreset(
        resize_max=preset.resize_max,
        denoise_strength=min(0.42, max(0.12, preset.denoise_strength + 0.06)),
        edge_weight=max(0.0, preset.edge_weight * 0.45),
        color_quant_k=max(24, int(preset.color_quant_k + 8)),
        contrast_gain=min(1.04, max(1.0, preset.contrast_gain * 0.96)),
        sharpen_amount=max(0.0, preset.sharpen_amount * 0.35),
        saturation_gain=min(1.04, max(1.0, preset.saturation_gain * 0.95)),
    )
