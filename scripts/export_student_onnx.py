from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.student_model import StudentCartoonizer


def _configure_console_encoding() -> None:
    # Prevent Windows cp1252 crashes from unicode exporter log messages.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_console_encoding()

    parser = argparse.ArgumentParser(description="Export distilled student model checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    width = int(checkpoint.get("width", 32))
    residual_blocks = int(checkpoint.get("residual_blocks", 4))

    model = StudentCartoonizer(width=width, residual_blocks=residual_blocks)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, args.image_size, args.image_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX model: {output_path}")


if __name__ == "__main__":
    main()
