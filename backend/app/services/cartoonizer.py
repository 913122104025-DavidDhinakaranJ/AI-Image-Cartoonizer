from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.services.style_registry import StyleRegistry

LOGGER = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


class CartoonizerService:
    def __init__(self, style_registry: StyleRegistry) -> None:
        self._style_registry = style_registry
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._warned_styles: set[str] = set()

    def cartoonize(self, image_bgr: np.ndarray, style_id: str) -> np.ndarray:
        session = self._load_session(style_id)
        if session is None:
            return self._fallback_cartoonize(image_bgr, style_id)

        return self._run_onnx_session(session, image_bgr)

    def _load_session(self, style_id: str):
        if style_id in self._sessions:
            return self._sessions[style_id]

        style = self._style_registry.get_style(style_id)
        if style is None:
            raise ValueError(f"Unknown style: {style_id}")

        model_path = style.model_path
        if model_path is None or not model_path.exists():
            self._warn_missing_model(style_id, model_path)
            return None
        if ort is None:
            self._warn_missing_model(style_id, model_path)
            return None
        if model_path.suffix.lower() != ".onnx":
            self._warn_missing_model(style_id, model_path)
            return None

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._sessions[style_id] = session
        LOGGER.info("Loaded ONNX model for style '%s' from %s", style_id, model_path)
        return session

    def _warn_missing_model(self, style_id: str, model_path: Path | None) -> None:
        if style_id in self._warned_styles:
            return
        self._warned_styles.add(style_id)
        LOGGER.warning(
            "Model not available for style '%s' (%s). Using fallback stylization.",
            style_id,
            model_path,
        )

    def _run_onnx_session(self, session, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        model_input = (rgb / 127.5) - 1.0

        input_meta = session.get_inputs()[0]
        input_shape = list(input_meta.shape)
        if len(input_shape) != 4:
            raise ValueError(f"Unsupported model input rank: {input_shape}")

        # Support both common ONNX image layouts:
        # - NHWC: [N, H, W, C]
        # - NCHW: [N, C, H, W]
        expects_nhwc = input_shape[-1] == 3
        expects_nchw = input_shape[1] == 3 if len(input_shape) > 1 else False

        if expects_nhwc:
            model_input = model_input[None, ...]
        elif expects_nchw:
            model_input = np.transpose(model_input, (2, 0, 1))[None, ...]
        else:
            raise ValueError(f"Unable to infer model input layout from shape: {input_shape}")

        input_name = input_meta.name
        output = session.run(None, {input_name: model_input})[0]

        if output.ndim == 4:
            output = output[0]
        if output.ndim != 3:
            raise ValueError(f"Unsupported model output shape: {output.shape}")

        if output.shape[-1] == 3:
            output = output
        elif output.shape[0] == 3:
            output = np.transpose(output, (1, 2, 0))
        else:
            raise ValueError(f"Unable to infer model output layout from shape: {output.shape}")

        if output.max() <= 1.5 and output.min() >= -1.5:
            output = (output + 1.0) / 2.0
            output = output * 255.0
        elif output.max() <= 1.5 and output.min() >= 0.0:
            output = output * 255.0

        output = np.clip(output, 0, 255).astype(np.uint8)
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    def _fallback_cartoonize(self, image_bgr: np.ndarray, style_id: str) -> np.ndarray:
        sigma_s = {"hayao": 65, "shinkai": 80, "paprika": 55}.get(style_id, 60)
        sigma_r = {"hayao": 0.45, "shinkai": 0.40, "paprika": 0.50}.get(style_id, 0.45)

        stylized = cv2.stylization(image_bgr, sigma_s=sigma_s, sigma_r=sigma_r)
        tinted = stylized.astype(np.int16)

        if style_id == "hayao":
            tinted[:, :, 1] += 8
            tinted[:, :, 2] += 6
        elif style_id == "shinkai":
            tinted[:, :, 0] += 10
            tinted[:, :, 2] += 4
        elif style_id == "paprika":
            tinted[:, :, 2] += 12
            tinted[:, :, 1] += 4

        return np.clip(tinted, 0, 255).astype(np.uint8)
