from __future__ import annotations

import cv2
import numpy as np


def _ssim(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def edge_ssim(input_bgr: np.ndarray, output_bgr: np.ndarray) -> float:
    input_gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    output_gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)
    if output_gray.shape != input_gray.shape:
        output_gray = cv2.resize(
            output_gray,
            (input_gray.shape[1], input_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    input_edges = cv2.Canny(input_gray, 90, 180)
    output_edges = cv2.Canny(output_gray, 90, 180)
    return _ssim(input_edges, output_edges)


def artifact_score(output_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.mean(np.abs(lap)) / 255.0)


def compute_metrics(input_bgr: np.ndarray, output_bgr: np.ndarray) -> dict[str, float]:
    return {
        "edge_ssim": round(edge_ssim(input_bgr, output_bgr), 4),
        "artifact_score": round(artifact_score(output_bgr), 4),
    }
