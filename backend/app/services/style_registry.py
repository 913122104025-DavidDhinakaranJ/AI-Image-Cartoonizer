from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AQEPreset:
    resize_max: int
    denoise_strength: float
    edge_weight: float
    color_quant_k: int
    contrast_gain: float
    sharpen_amount: float
    saturation_gain: float


@dataclass(frozen=True)
class StyleConfig:
    id: str
    name: str
    preview: str
    model_path: Path | None
    aqe: AQEPreset


class StyleRegistry:
    def __init__(self, presets_path: Path) -> None:
        self._presets_path = presets_path
        self._styles: dict[str, StyleConfig] = {}
        self._load()

    def _load(self) -> None:
        if not self._presets_path.exists():
            raise FileNotFoundError(f"Missing style preset file: {self._presets_path}")

        payload = json.loads(self._presets_path.read_text(encoding="utf-8"))
        styles = payload.get("styles", [])
        if not styles:
            raise ValueError("style_presets.json must define at least one style")

        root = self._presets_path.parent
        for row in styles:
            model_path_value = row.get("model_path")
            model_path = root / model_path_value if model_path_value else None

            aqe_row = row.get("aqe", {})
            aqe = AQEPreset(
                resize_max=int(aqe_row.get("resize_max", 1024)),
                denoise_strength=float(aqe_row.get("denoise_strength", 0.2)),
                edge_weight=float(aqe_row.get("edge_weight", 0.2)),
                color_quant_k=int(aqe_row.get("color_quant_k", 24)),
                contrast_gain=float(aqe_row.get("contrast_gain", 1.05)),
                sharpen_amount=float(aqe_row.get("sharpen_amount", 0.2)),
                saturation_gain=float(aqe_row.get("saturation_gain", 1.08)),
            )

            style = StyleConfig(
                id=row["id"],
                name=row.get("name", row["id"].capitalize()),
                preview=row.get("preview", ""),
                model_path=model_path,
                aqe=aqe,
            )
            self._styles[style.id] = style

    def list_styles(self) -> list[StyleConfig]:
        return list(self._styles.values())

    def get_style(self, style_id: str) -> StyleConfig | None:
        return self._styles.get(style_id)

    def get_preset(self, style_id: str) -> AQEPreset:
        style = self.get_style(style_id)
        if style is None:
            raise KeyError(f"Unknown style: {style_id}")
        return style.aqe
