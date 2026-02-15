from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


class ResultStore:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, image_bgr: np.ndarray, style_id: str, variant: str) -> str:
        filename = f"{uuid4().hex}_{style_id}_{variant}.png"
        output_path = self._output_dir / filename
        success = cv2.imwrite(str(output_path), image_bgr)
        if not success:
            raise RuntimeError("Failed to write output image.")
        return filename
