from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import httpx


def iter_images(path: Path) -> Iterable[Path]:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        yield from path.rglob(f"*{suffix}")


def mime_for(path: Path) -> str:
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    return mapping.get(path.suffix.lower(), "application/octet-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create teacher-student training pairs.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--input-dir", required=True, help="Source real images folder")
    parser.add_argument("--output-dir", default="training/distill_data")
    parser.add_argument("--style-id", required=True, choices=["hayao", "shinkai", "paprika"])
    parser.add_argument("--variant", default="improved_lite", choices=["improved", "improved_lite"])
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    base_output = Path(args.output_dir) / args.style_id
    inputs_dir = base_output / "inputs"
    targets_dir = base_output / "targets"
    manifest_path = base_output / "manifest.csv"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(iter_images(input_dir))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")

    rows: list[dict[str, str]] = []
    with httpx.Client(timeout=300.0) as client:
        for idx, image_path in enumerate(image_paths, start=1):
            with image_path.open("rb") as handle:
                files = {"image": (image_path.name, handle, mime_for(image_path))}
                data = {"style_id": args.style_id, "variant": args.variant}
                response = client.post(f"{args.api_url}/api/cartoonize", files=files, data=data)
            response.raise_for_status()
            payload = response.json()

            result_url = payload["result_url"]
            result_response = client.get(f"{args.api_url}{result_url}")
            result_response.raise_for_status()

            input_name = f"{idx:05d}{image_path.suffix.lower()}"
            target_name = f"{idx:05d}.png"
            input_dst = inputs_dir / input_name
            target_dst = targets_dir / target_name

            input_dst.write_bytes(image_path.read_bytes())
            target_dst.write_bytes(result_response.content)

            rows.append(
                {
                    "input_path": str(input_dst.as_posix()),
                    "target_path": str(target_dst.as_posix()),
                    "style_id": args.style_id,
                }
            )
            print(f"[{idx}/{len(image_paths)}] paired {image_path.name}")

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input_path", "target_path", "style_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} pairs to {base_output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
