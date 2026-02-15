from __future__ import annotations

import numpy as np

from app.services.postprocess import postprocess_improved
from app.services.preprocess import preprocess_baseline, preprocess_improved
from app.services.style_registry import AQEPreset


def _preset() -> AQEPreset:
    return AQEPreset(
        resize_max=256,
        denoise_strength=0.25,
        edge_weight=0.2,
        color_quant_k=16,
        contrast_gain=1.05,
        sharpen_amount=0.2,
        saturation_gain=1.1,
    )


def test_preprocess_resizes_to_max_side() -> None:
    image = np.full((800, 1200, 3), 180, dtype=np.uint8)
    output = preprocess_baseline(image, max_side=256)
    assert max(output.shape[:2]) == 256


def test_improved_pipeline_preserves_shape() -> None:
    image = np.full((300, 400, 3), 100, dtype=np.uint8)
    preset = _preset()
    processed = preprocess_improved(image, preset)
    styled = np.full_like(processed, 150)
    output = postprocess_improved(styled, processed, preset)
    assert output.shape == processed.shape
    assert output.dtype == np.uint8


def test_improved_pipeline_handles_shape_mismatch() -> None:
    source = np.full((175, 289, 3), 120, dtype=np.uint8)
    stylized = np.full((172, 288, 3), 160, dtype=np.uint8)
    output = postprocess_improved(stylized, source, _preset())
    assert output.shape == source.shape
