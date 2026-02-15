from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Iterable

import httpx


def iter_images(path: pathlib.Path) -> Iterable[pathlib.Path]:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        yield from path.glob(f"*{suffix}")


def call_variant(api_url: str, image_path: pathlib.Path, style_id: str, variant: str) -> dict:
    mime_by_suffix = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime = mime_by_suffix.get(image_path.suffix.lower(), "application/octet-stream")
    with image_path.open("rb") as handle:
        files = {"image": (image_path.name, handle, mime)}
        data = {"style_id": style_id, "variant": variant}
        response = httpx.post(f"{api_url}/api/cartoonize", files=files, data=data, timeout=180)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs improved cartoonization.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", default="evaluation/results.csv")
    parser.add_argument("--styles", default="hayao,shinkai,paprika")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_csv = pathlib.Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    styles = [value.strip() for value in args.styles.split(",") if value.strip()]

    rows: list[dict] = []
    for image_path in iter_images(input_dir):
        for style_id in styles:
            baseline = call_variant(args.api_url, image_path, style_id, "baseline")
            improved = call_variant(args.api_url, image_path, style_id, "improved")
            rows.append(
                {
                    "image": image_path.name,
                    "style_id": style_id,
                    "baseline_edge_ssim": baseline["metrics"]["edge_ssim"],
                    "improved_edge_ssim": improved["metrics"]["edge_ssim"],
                    "baseline_artifact_score": baseline["metrics"]["artifact_score"],
                    "improved_artifact_score": improved["metrics"]["artifact_score"],
                    "baseline_latency_ms": baseline["latency_ms"],
                    "improved_latency_ms": improved["latency_ms"],
                }
            )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
