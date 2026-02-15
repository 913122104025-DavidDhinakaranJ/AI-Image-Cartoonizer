from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _make_image_bytes() -> bytes:
    image = Image.new("RGB", (320, 240), color=(120, 180, 220))
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    return payload.getvalue()


def test_get_styles_returns_three_styles() -> None:
    response = client.get("/api/styles")
    assert response.status_code == 200
    body = response.json()
    assert "styles" in body
    assert len(body["styles"]) == 3


def test_cartoonize_baseline_and_improved() -> None:
    image_bytes = _make_image_bytes()
    files = {"image": ("sample.png", image_bytes, "image/png")}

    baseline = client.post(
        "/api/cartoonize",
        files=files,
        data={"style_id": "hayao", "variant": "baseline"},
    )
    assert baseline.status_code == 200
    baseline_body = baseline.json()
    assert baseline_body["variant"] == "baseline"
    assert baseline_body["style_id"] == "hayao"
    assert baseline_body["result_url"].startswith("/api/results/")

    improved = client.post(
        "/api/cartoonize",
        files=files,
        data={"style_id": "hayao", "variant": "improved"},
    )
    assert improved.status_code == 200
    improved_body = improved.json()
    assert improved_body["variant"] == "improved"
    assert improved_body["metrics"]["edge_ssim"] >= -1.0
    assert improved_body["metrics"]["artifact_score"] >= 0.0


def test_cartoonize_rejects_unsupported_file_type() -> None:
    files = {"image": ("bad.txt", b"not-image", "text/plain")}
    response = client.post(
        "/api/cartoonize",
        files=files,
        data={"style_id": "hayao", "variant": "improved"},
    )
    assert response.status_code == 415
