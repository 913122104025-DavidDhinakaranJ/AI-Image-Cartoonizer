from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class StudentCartoonizer(nn.Module):
    """Small U-Net style network for teacher-student distillation."""

    def __init__(self, width: int = 32, residual_blocks: int = 4) -> None:
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        for _ in range(residual_blocks):
            blocks.append(ResidualBlock(width * 4))
        self.bottleneck = nn.Sequential(*blocks)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(width * 2, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.dec3(b)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        return self.out(d2)

