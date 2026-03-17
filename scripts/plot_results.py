from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def plot_summary(summary_rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    rows = [row for row in summary_rows if row.get("style_id") not in (None, "__all__")]
    if not rows:
        return []

    styles = [row["style_id"] for row in rows]
    baseline_edge = [to_float(row.get("baseline_edge_ssim_mean", "0")) for row in rows]
    improved_edge = [to_float(row.get("improved_edge_ssim_mean", "0")) for row in rows]
    baseline_artifact = [to_float(row.get("baseline_artifact_mean", "0")) for row in rows]
    improved_artifact = [to_float(row.get("improved_artifact_mean", "0")) for row in rows]
    baseline_latency = [to_float(row.get("baseline_latency_ms_mean", "0")) for row in rows]
    improved_latency = [to_float(row.get("improved_latency_ms_mean", "0")) for row in rows]
    edge_win = [to_float(row.get("edge_win_rate", "0")) for row in rows]
    artifact_win = [to_float(row.get("artifact_win_rate", "0")) for row in rows]
    overall_win = [to_float(row.get("overall_win_rate", "0")) for row in rows]

    paths: list[Path] = []

    # Chart 1: quality bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = range(len(styles))
    width = 0.36
    axes[0].bar([i - width / 2 for i in x], baseline_edge, width=width, label="Baseline")
    axes[0].bar([i + width / 2 for i in x], improved_edge, width=width, label="Improved")
    axes[0].set_title("Edge SSIM (higher is better)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(styles)
    axes[0].legend()

    axes[1].bar([i - width / 2 for i in x], baseline_artifact, width=width, label="Baseline")
    axes[1].bar([i + width / 2 for i in x], improved_artifact, width=width, label="Improved")
    axes[1].set_title("Artifact Score (lower is better)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(styles)
    axes[1].legend()

    fig.tight_layout()
    quality_path = output_dir / "quality_comparison.png"
    fig.savefig(quality_path, dpi=180)
    plt.close(fig)
    paths.append(quality_path)

    # Chart 2: latency bars
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar([i - width / 2 for i in x], baseline_latency, width=width, label="Baseline")
    ax.bar([i + width / 2 for i in x], improved_latency, width=width, label="Improved")
    ax.set_title("Latency Comparison (ms)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(styles)
    ax.legend()
    fig.tight_layout()
    latency_path = output_dir / "latency_comparison.png"
    fig.savefig(latency_path, dpi=180)
    plt.close(fig)
    paths.append(latency_path)

    # Chart 3: win rates
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(styles, edge_win, marker="o", label="Edge win rate")
    ax.plot(styles, artifact_win, marker="o", label="Artifact win rate")
    ax.plot(styles, overall_win, marker="o", label="Overall win rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Improved Variant Win Rates")
    ax.set_ylabel("Rate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    win_rate_path = output_dir / "win_rates.png"
    fig.savefig(win_rate_path, dpi=180)
    plt.close(fig)
    paths.append(win_rate_path)

    return paths


def plot_tuning(trial_rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    if not trial_rows:
        return []

    style_groups: dict[str, list[dict[str, str]]] = {}
    for row in trial_rows:
        style_groups.setdefault(row.get("style_id", "unknown"), []).append(row)

    paths: list[Path] = []
    for style_id, rows in style_groups.items():
        rows = sorted(rows, key=lambda r: int(to_float(r.get("trial_index", "0"))))
        trial_index = [int(to_float(r.get("trial_index", "0"))) for r in rows]
        score = [to_float(r.get("score", "0")) for r in rows]
        edge_gain = [to_float(r.get("edge_gain", "0")) for r in rows]
        artifact_gain = [to_float(r.get("artifact_gain", "0")) for r in rows]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].plot(trial_index, score, marker="o")
        axes[0].set_title(f"{style_id} tuning score by trial")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Score")
        axes[0].grid(True, alpha=0.25)

        axes[1].scatter(edge_gain, artifact_gain, alpha=0.8)
        axes[1].set_title(f"{style_id} gains (edge vs artifact)")
        axes[1].set_xlabel("Edge gain")
        axes[1].set_ylabel("Artifact gain")
        axes[1].axhline(0, color="gray", linewidth=1)
        axes[1].axvline(0, color="gray", linewidth=1)
        axes[1].grid(True, alpha=0.25)

        fig.tight_layout()
        path = output_dir / f"tuning_{style_id}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report-ready plots from evaluation CSV files.")
    parser.add_argument("--summary-csv", default="evaluation/summary.csv")
    parser.add_argument("--trials-csv", default="evaluation/tuning_trials.csv")
    parser.add_argument("--output-dir", default="evaluation/plots")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    trials_csv = Path(args.trials_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    generated.extend(plot_summary(read_csv(summary_csv), output_dir))
    generated.extend(plot_tuning(read_csv(trials_csv), output_dir))

    if not generated:
        raise SystemExit(
            "No plots were generated. Ensure summary/trials CSV files exist and contain rows."
        )

    print("Generated plots:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
