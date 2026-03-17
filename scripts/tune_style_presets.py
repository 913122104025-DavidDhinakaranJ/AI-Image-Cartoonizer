from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.cartoonizer import CartoonizerService
from app.services.metrics import compute_metrics
from app.services.postprocess import postprocess_improved
from app.services.preprocess import preprocess_baseline, preprocess_improved
from app.services.style_registry import AQEPreset, StyleRegistry


def iter_images(path: Path) -> Iterable[Path]:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        yield from path.rglob(f"*{suffix}")


def load_images(input_dir: Path, max_images: int) -> list[np.ndarray]:
    image_paths = sorted(iter_images(input_dir))
    if max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    images: list[np.ndarray] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        images.append(image)
    if not images:
        raise ValueError(f"No decodable images found in {input_dir}")
    return images


def ensure_shape(output_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
    if output_bgr.shape[:2] == reference_bgr.shape[:2]:
        return output_bgr
    return cv2.resize(
        output_bgr,
        (reference_bgr.shape[1], reference_bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def parse_float_values(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_values(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def evaluate_baseline(
    images: list[np.ndarray],
    style_id: str,
    cartoonizer: CartoonizerService,
    resize_max: int,
) -> dict[str, float]:
    edge_scores: list[float] = []
    artifact_scores: list[float] = []
    latencies: list[float] = []

    for image in images:
        prepared = preprocess_baseline(image, resize_max)
        started = time.perf_counter()
        output = cartoonizer.cartoonize(prepared, style_id)
        output = ensure_shape(output, prepared)
        latency_ms = (time.perf_counter() - started) * 1000.0
        metrics = compute_metrics(prepared, output)
        edge_scores.append(float(metrics["edge_ssim"]))
        artifact_scores.append(float(metrics["artifact_score"]))
        latencies.append(latency_ms)

    return {
        "edge_ssim_mean": fmean(edge_scores),
        "artifact_score_mean": fmean(artifact_scores),
        "latency_ms_mean": fmean(latencies),
    }


def evaluate_candidate(
    images: list[np.ndarray],
    style_id: str,
    cartoonizer: CartoonizerService,
    preset: AQEPreset,
) -> dict[str, float]:
    edge_scores: list[float] = []
    artifact_scores: list[float] = []
    latencies: list[float] = []

    for image in images:
        prepared = preprocess_improved(image, preset)
        started = time.perf_counter()
        stylized = cartoonizer.cartoonize(prepared, style_id)
        output = postprocess_improved(stylized, prepared, preset)
        output = ensure_shape(output, prepared)
        latency_ms = (time.perf_counter() - started) * 1000.0
        metrics = compute_metrics(prepared, output)
        edge_scores.append(float(metrics["edge_ssim"]))
        artifact_scores.append(float(metrics["artifact_score"]))
        latencies.append(latency_ms)

    return {
        "edge_ssim_mean": fmean(edge_scores),
        "artifact_score_mean": fmean(artifact_scores),
        "latency_ms_mean": fmean(latencies),
    }


def build_candidates(
    current: AQEPreset,
    denoise_values: list[float],
    edge_values: list[float],
    k_values: list[int],
    contrast_values: list[float],
    sharpen_values: list[float],
    saturation_values: list[float],
    max_trials: int,
    seed: int,
) -> list[AQEPreset]:
    combos = list(
        itertools.product(
            denoise_values,
            edge_values,
            k_values,
            contrast_values,
            sharpen_values,
            saturation_values,
        )
    )

    rng = random.Random(seed)
    if max_trials > 0 and len(combos) > max_trials - 1:
        sampled = rng.sample(combos, max_trials - 1)
    else:
        sampled = combos

    candidates = [
        AQEPreset(
            resize_max=current.resize_max,
            denoise_strength=current.denoise_strength,
            edge_weight=current.edge_weight,
            color_quant_k=current.color_quant_k,
            contrast_gain=current.contrast_gain,
            sharpen_amount=current.sharpen_amount,
            saturation_gain=current.saturation_gain,
        )
    ]
    for combo in sampled:
        candidate = AQEPreset(
            resize_max=current.resize_max,
            denoise_strength=float(combo[0]),
            edge_weight=float(combo[1]),
            color_quant_k=int(combo[2]),
            contrast_gain=float(combo[3]),
            sharpen_amount=float(combo[4]),
            saturation_gain=float(combo[5]),
        )
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def apply_best_presets(style_presets_path: Path, tuned: dict[str, AQEPreset]) -> None:
    payload = json.loads(style_presets_path.read_text(encoding="utf-8"))
    for style in payload.get("styles", []):
        style_id = style.get("id")
        if style_id not in tuned:
            continue
        preset = tuned[style_id]
        style["aqe"] = {
            "resize_max": int(preset.resize_max),
            "denoise_strength": round(float(preset.denoise_strength), 4),
            "edge_weight": round(float(preset.edge_weight), 4),
            "color_quant_k": int(preset.color_quant_k),
            "contrast_gain": round(float(preset.contrast_gain), 4),
            "sharpen_amount": round(float(preset.sharpen_amount), 4),
            "saturation_gain": round(float(preset.saturation_gain), 4),
        }
    style_presets_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-tune AQE style presets.")
    parser.add_argument("--input-dir", required=True, help="Folder of evaluation images")
    parser.add_argument("--styles", default="hayao,shinkai,paprika")
    parser.add_argument("--max-images", type=int, default=24, help="0 means all images")
    parser.add_argument("--max-trials", type=int, default=20, help="Per style, includes current preset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-weight", type=float, default=0.45)
    parser.add_argument("--artifact-weight", type=float, default=2.2)
    parser.add_argument("--latency-weight", type=float, default=0.35)
    parser.add_argument("--artifact-regression-penalty", type=float, default=2.0)
    parser.add_argument("--denoise-values", default="0.24,0.28,0.32,0.36")
    parser.add_argument("--edge-values", default="0.06,0.10,0.14,0.18")
    parser.add_argument("--k-values", default="24,28,32,36")
    parser.add_argument("--contrast-values", default="1.00,1.02,1.04")
    parser.add_argument("--sharpen-values", default="0.00,0.06,0.10")
    parser.add_argument("--saturation-values", default="1.00,1.03,1.06")
    parser.add_argument("--style-presets", default="backend/style_presets.json")
    parser.add_argument("--trials-csv", default="evaluation/tuning_trials.csv")
    parser.add_argument("--best-json", default="evaluation/tuned_presets.json")
    parser.add_argument("--apply", action="store_true", help="Write best presets back to style_presets.json")
    args = parser.parse_args()

    style_presets_path = (REPO_ROOT / args.style_presets).resolve()
    if not style_presets_path.exists():
        raise SystemExit(f"Missing style presets file: {style_presets_path}")

    images = load_images(Path(args.input_dir), args.max_images)
    style_registry = StyleRegistry(style_presets_path)
    cartoonizer = CartoonizerService(style_registry)
    styles = [value.strip() for value in args.styles.split(",") if value.strip()]

    denoise_values = parse_float_values(args.denoise_values)
    edge_values = parse_float_values(args.edge_values)
    k_values = parse_int_values(args.k_values)
    contrast_values = parse_float_values(args.contrast_values)
    sharpen_values = parse_float_values(args.sharpen_values)
    saturation_values = parse_float_values(args.saturation_values)

    trial_rows: list[dict[str, float | int | str]] = []
    best_presets: dict[str, AQEPreset] = {}
    best_payload: dict[str, dict[str, float | int]] = {}

    for style_id in styles:
        current = style_registry.get_preset(style_id)
        baseline = evaluate_baseline(images, style_id, cartoonizer, current.resize_max)
        style_seed = args.seed + sum(ord(ch) for ch in style_id)
        candidates = build_candidates(
            current=current,
            denoise_values=denoise_values,
            edge_values=edge_values,
            k_values=k_values,
            contrast_values=contrast_values,
            sharpen_values=sharpen_values,
            saturation_values=saturation_values,
            max_trials=max(1, args.max_trials),
            seed=style_seed,
        )

        best_score = float("-inf")
        best_candidate = current
        print(f"Tuning style '{style_id}' with {len(candidates)} candidates...")

        for idx, candidate in enumerate(candidates, start=1):
            measured = evaluate_candidate(images, style_id, cartoonizer, candidate)
            edge_gain = measured["edge_ssim_mean"] - baseline["edge_ssim_mean"]
            artifact_gain = baseline["artifact_score_mean"] - measured["artifact_score_mean"]
            latency_penalty = max(
                0.0,
                measured["latency_ms_mean"] - baseline["latency_ms_mean"],
            ) / 1000.0
            artifact_component = args.artifact_weight * artifact_gain
            if artifact_gain < 0:
                artifact_component *= args.artifact_regression_penalty
            score = (args.edge_weight * edge_gain) + artifact_component - (
                args.latency_weight * latency_penalty
            )

            trial_rows.append(
                {
                    "style_id": style_id,
                    "trial_index": idx,
                    "score": round(score, 6),
                    "edge_gain": round(edge_gain, 6),
                    "artifact_gain": round(artifact_gain, 6),
                    "latency_penalty_s": round(latency_penalty, 6),
                    "baseline_edge_ssim_mean": round(baseline["edge_ssim_mean"], 6),
                    "candidate_edge_ssim_mean": round(measured["edge_ssim_mean"], 6),
                    "baseline_artifact_mean": round(baseline["artifact_score_mean"], 6),
                    "candidate_artifact_mean": round(measured["artifact_score_mean"], 6),
                    "baseline_latency_ms_mean": round(baseline["latency_ms_mean"], 2),
                    "candidate_latency_ms_mean": round(measured["latency_ms_mean"], 2),
                    "resize_max": candidate.resize_max,
                    "denoise_strength": candidate.denoise_strength,
                    "edge_weight": candidate.edge_weight,
                    "color_quant_k": candidate.color_quant_k,
                    "contrast_gain": candidate.contrast_gain,
                    "sharpen_amount": candidate.sharpen_amount,
                    "saturation_gain": candidate.saturation_gain,
                }
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate
            print(
                f"  Trial {idx}/{len(candidates)} "
                f"score={score:.5f} edge_gain={edge_gain:.5f} artifact_gain={artifact_gain:.5f}"
            )

        best_presets[style_id] = best_candidate
        best_payload[style_id] = {
            "score": round(best_score, 6),
            **asdict(best_candidate),
        }
        print(f"Best preset for '{style_id}': {best_payload[style_id]}")

    trials_csv_path = (REPO_ROOT / args.trials_csv).resolve()
    best_json_path = (REPO_ROOT / args.best_json).resolve()
    write_csv(trials_csv_path, trial_rows)
    best_json_path.parent.mkdir(parents=True, exist_ok=True)
    best_json_path.write_text(json.dumps(best_payload, indent=2) + "\n", encoding="utf-8")

    if args.apply:
        apply_best_presets(style_presets_path, best_presets)
        print(f"Applied tuned presets to {style_presets_path}")

    print(f"Wrote trial results to {trials_csv_path}")
    print(f"Wrote best presets to {best_json_path}")


if __name__ == "__main__":
    main()
