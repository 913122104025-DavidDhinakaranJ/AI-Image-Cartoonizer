from __future__ import annotations

import time
from pathlib import Path

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import (
    CartoonizeResponse,
    HealthResponse,
    MetricPayload,
    StyleInfo,
    StylesResponse,
)
from app.services.cartoonizer import CartoonizerService
from app.services.metrics import compute_metrics
from app.services.postprocess import postprocess_improved
from app.services.preprocess import decode_image, preprocess_baseline, preprocess_improved
from app.services.result_store import ResultStore
from app.services.style_registry import StyleRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"
STYLE_PRESETS_PATH = BACKEND_ROOT / "style_presets.json"
RESULTS_DIR = BACKEND_ROOT / "results"

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}

style_registry = StyleRegistry(STYLE_PRESETS_PATH)
cartoonizer_service = CartoonizerService(style_registry)
result_store = ResultStore(RESULTS_DIR)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/styles", response_model=StylesResponse)
def get_styles() -> StylesResponse:
    styles = [
        StyleInfo(id=style.id, name=style.name, preview=style.preview)
        for style in style_registry.list_styles()
    ]
    return StylesResponse(styles=styles)


@router.post("/cartoonize", response_model=CartoonizeResponse)
async def cartoonize_image(
    image: UploadFile = File(...),
    style_id: str = Form(...),
    variant: str = Form("improved"),
) -> CartoonizeResponse:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file format. Use JPG, PNG, or WEBP.",
        )

    if variant not in {"baseline", "improved"}:
        raise HTTPException(status_code=400, detail="variant must be baseline or improved")

    style = style_registry.get_style(style_id)
    if style is None:
        raise HTTPException(status_code=400, detail=f"Unknown style_id: {style_id}")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        decoded = decode_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    preset = style_registry.get_preset(style_id)
    started = time.perf_counter()

    if variant == "baseline":
        prepared = preprocess_baseline(decoded, preset.resize_max)
        stylized = cartoonizer_service.cartoonize(prepared, style_id)
        output = stylized
    else:
        prepared = preprocess_improved(decoded, preset)
        stylized = cartoonizer_service.cartoonize(prepared, style_id)
        output = postprocess_improved(stylized, prepared, preset)

    if output.shape[:2] != prepared.shape[:2]:
        output = cv2.resize(output, (prepared.shape[1], prepared.shape[0]), interpolation=cv2.INTER_LINEAR)

    metrics = compute_metrics(prepared, output)
    filename = result_store.save(output, style_id, variant)
    latency_ms = int((time.perf_counter() - started) * 1000)

    return CartoonizeResponse(
        result_url=f"/api/results/{filename}",
        style_id=style_id,
        variant=variant,
        latency_ms=latency_ms,
        metrics=MetricPayload(**metrics),
    )
