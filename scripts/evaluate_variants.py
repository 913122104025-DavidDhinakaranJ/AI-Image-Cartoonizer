from __future__ import annotations

import argparse
import csv
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import fmean
from typing import Iterable

import httpx


def iter_images(path: pathlib.Path) -> Iterable[pathlib.Path]:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        yield from path.rglob(f"*{suffix}")


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


def evaluate_pair(
    api_url: str,
    image_path: pathlib.Path,
    style_id: str,
    compare_variant: str,
) -> dict[str, float | str]:
    baseline = call_variant(api_url, image_path, style_id, "baseline")
    improved = call_variant(api_url, image_path, style_id, compare_variant)

    edge_delta = improved["metrics"]["edge_ssim"] - baseline["metrics"]["edge_ssim"]
    artifact_delta = improved["metrics"]["artifact_score"] - baseline["metrics"]["artifact_score"]
    latency_delta_ms = improved["latency_ms"] - baseline["latency_ms"]
    return {
        "image": image_path.name,
        "style_id": style_id,
        "baseline_edge_ssim": baseline["metrics"]["edge_ssim"],
        "improved_edge_ssim": improved["metrics"]["edge_ssim"],
        "edge_ssim_delta": round(edge_delta, 4),
        "baseline_artifact_score": baseline["metrics"]["artifact_score"],
        "improved_artifact_score": improved["metrics"]["artifact_score"],
        "artifact_score_delta": round(artifact_delta, 4),
        "baseline_latency_ms": baseline["latency_ms"],
        "improved_latency_ms": improved["latency_ms"],
        "latency_delta_ms": latency_delta_ms,
    }


def build_summary(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    if not rows:
        return []

    grouped: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        style = str(row["style_id"])
        grouped.setdefault(style, []).append(row)

    summary: list[dict[str, float | str]] = []
    for style_id, entries in grouped.items():
        baseline_edge = [float(item["baseline_edge_ssim"]) for item in entries]
        improved_edge = [float(item["improved_edge_ssim"]) for item in entries]
        baseline_artifact = [float(item["baseline_artifact_score"]) for item in entries]
        improved_artifact = [float(item["improved_artifact_score"]) for item in entries]
        baseline_latency = [float(item["baseline_latency_ms"]) for item in entries]
        improved_latency = [float(item["improved_latency_ms"]) for item in entries]

        edge_delta = [imp - base for base, imp in zip(baseline_edge, improved_edge)]
        artifact_delta = [imp - base for base, imp in zip(baseline_artifact, improved_artifact)]
        latency_delta = [imp - base for base, imp in zip(baseline_latency, improved_latency)]

        edge_win_rate = sum(1 for value in edge_delta if value > 0) / len(edge_delta)
        artifact_win_rate = sum(1 for value in artifact_delta if value < 0) / len(artifact_delta)
        overall_win_rate = (
            sum(
                1
                for e_delta, a_delta in zip(edge_delta, artifact_delta)
                if e_delta >= 0 and a_delta <= 0
            )
            / len(edge_delta)
        )

        summary.append(
            {
                "style_id": style_id,
                "samples": len(entries),
                "baseline_edge_ssim_mean": round(fmean(baseline_edge), 4),
                "improved_edge_ssim_mean": round(fmean(improved_edge), 4),
                "edge_ssim_delta_mean": round(fmean(edge_delta), 4),
                "baseline_artifact_mean": round(fmean(baseline_artifact), 4),
                "improved_artifact_mean": round(fmean(improved_artifact), 4),
                "artifact_delta_mean": round(fmean(artifact_delta), 4),
                "baseline_latency_ms_mean": round(fmean(baseline_latency), 2),
                "improved_latency_ms_mean": round(fmean(improved_latency), 2),
                "latency_delta_ms_mean": round(fmean(latency_delta), 2),
                "edge_win_rate": round(edge_win_rate, 4),
                "artifact_win_rate": round(artifact_win_rate, 4),
                "overall_win_rate": round(overall_win_rate, 4),
            }
        )

    if summary:
        all_rows = list(rows)
        summary.append(
            {
                "style_id": "__all__",
                "samples": len(all_rows),
                "baseline_edge_ssim_mean": round(
                    fmean(float(item["baseline_edge_ssim"]) for item in all_rows), 4
                ),
                "improved_edge_ssim_mean": round(
                    fmean(float(item["improved_edge_ssim"]) for item in all_rows), 4
                ),
                "edge_ssim_delta_mean": round(
                    fmean(float(item["edge_ssim_delta"]) for item in all_rows), 4
                ),
                "baseline_artifact_mean": round(
                    fmean(float(item["baseline_artifact_score"]) for item in all_rows), 4
                ),
                "improved_artifact_mean": round(
                    fmean(float(item["improved_artifact_score"]) for item in all_rows), 4
                ),
                "artifact_delta_mean": round(
                    fmean(float(item["artifact_score_delta"]) for item in all_rows), 4
                ),
                "baseline_latency_ms_mean": round(
                    fmean(float(item["baseline_latency_ms"]) for item in all_rows), 2
                ),
                "improved_latency_ms_mean": round(
                    fmean(float(item["improved_latency_ms"]) for item in all_rows), 2
                ),
                "latency_delta_ms_mean": round(
                    fmean(float(item["latency_delta_ms"]) for item in all_rows), 2
                ),
                "edge_win_rate": round(
                    sum(1 for item in all_rows if float(item["edge_ssim_delta"]) > 0) / len(all_rows),
                    4,
                ),
                "artifact_win_rate": round(
                    sum(
                        1
                        for item in all_rows
                        if float(item["artifact_score_delta"]) < 0
                    )
                    / len(all_rows),
                    4,
                ),
                "overall_win_rate": round(
                    sum(
                        1
                        for item in all_rows
                        if float(item["edge_ssim_delta"]) >= 0
                        and float(item["artifact_score_delta"]) <= 0
                    )
                    / len(all_rows),
                    4,
                ),
            }
        )

    return summary


def write_csv(path: pathlib.Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs improved cartoonization.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", default="evaluation/results.csv")
    parser.add_argument("--summary-csv", default="evaluation/summary.csv")
    parser.add_argument("--styles", default="hayao,shinkai,paprika")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--workers", type=int, default=3, help="Parallel image-style jobs")
    parser.add_argument(
        "--compare-variant",
        choices=["improved", "improved_lite"],
        default="improved_lite",
        help="Variant compared against baseline",
    )
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_csv = pathlib.Path(args.output_csv)
    summary_csv = pathlib.Path(args.summary_csv)
    styles = [value.strip() for value in args.styles.split(",") if value.strip()]

    image_paths = sorted(iter_images(input_dir))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")

    jobs = [(image_path, style_id) for image_path in image_paths for style_id in styles]
    workers = max(1, args.workers)
    rows: list[dict[str, float | str]] = []

    if workers == 1:
        for image_path, style_id in jobs:
            row = evaluate_pair(args.api_url, image_path, style_id, args.compare_variant)
            rows.append(row)
            print(f"Evaluated {image_path.name} [{style_id}] vs {args.compare_variant}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    evaluate_pair,
                    args.api_url,
                    image_path,
                    style_id,
                    args.compare_variant,
                ): (image_path, style_id)
                for image_path, style_id in jobs
            }
            for future in as_completed(future_map):
                image_path, style_id = future_map[future]
                row = future.result()
                rows.append(row)
                print(f"Evaluated {image_path.name} [{style_id}] vs {args.compare_variant}")

    rows.sort(key=lambda item: (str(item["image"]), str(item["style_id"])))
    summary_rows = build_summary(rows)
    write_csv(output_csv, rows)
    write_csv(summary_csv, summary_rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")
    print(f"Wrote {len(summary_rows)} summary rows to {summary_csv}")


if __name__ == "__main__":
    main()
