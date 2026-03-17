from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.student_model import StudentCartoonizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml  # type: ignore

        return torch_directml.device()
    except Exception:  # noqa: BLE001
        return torch.device("cpu")


def image_to_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class PairDataset(Dataset):
    def __init__(self, manifest_path: Path, image_size: int) -> None:
        self.image_size = image_size
        self.rows: list[tuple[Path, Path]] = []
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                self.rows.append((Path(row["input_path"]), Path(row["target_path"])))
        if not self.rows:
            raise ValueError(f"No rows found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_path, target_path = self.rows[index]
        return image_to_tensor(input_path, self.image_size), image_to_tensor(target_path, self.image_size)


def edge_map(x: torch.Tensor) -> torch.Tensor:
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(1)
    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(1)
    gx = torch.nn.functional.conv2d(gray, kernel_x, padding=1)
    gy = torch.nn.functional.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight student cartoonizer via distillation.")
    parser.add_argument("--manifest", required=True, help="Path to pair manifest CSV")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--edge-loss-weight", type=float, default=0.15)
    parser.add_argument("--out-dir", default="training/artifacts")
    parser.add_argument("--style-id", required=True, choices=["hayao", "shinkai", "paprika"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PairDataset(manifest_path, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = pick_device()
    model = StudentCartoonizer(width=args.width, residual_blocks=args.residual_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    l1 = nn.L1Loss()

    metrics_rows: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_path = out_dir / f"{args.style_id}_student_best.pt"

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Training on device: {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_edge = 0.0
        batch_count = 0

        for inputs, targets in loader:
            batch_count += 1
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(inputs)
                recon_loss = l1(preds, targets)
                e_loss = l1(edge_map(preds), edge_map(targets))
                loss = recon_loss + (args.edge_loss_weight * e_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            total_recon += float(recon_loss.item())
            total_edge += float(e_loss.item())

        epoch_loss = total_loss / max(1, batch_count)
        epoch_recon = total_recon / max(1, batch_count)
        epoch_edge = total_edge / max(1, batch_count)

        row = {
            "epoch": epoch,
            "loss": round(epoch_loss, 6),
            "recon_loss": round(epoch_recon, 6),
            "edge_loss": round(epoch_edge, 6),
        }
        metrics_rows.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"loss={epoch_loss:.6f} recon={epoch_recon:.6f} edge={epoch_edge:.6f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "style_id": args.style_id,
                "model_state_dict": model.state_dict(),
                "width": args.width,
                "residual_blocks": args.residual_blocks,
                "image_size": args.image_size,
                "best_loss": best_loss,
            }
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")

    metrics_path = out_dir / f"{args.style_id}_training_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "recon_loss", "edge_loss"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"Training complete. Best loss={best_loss:.6f}")
    print(f"Checkpoint: {best_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
