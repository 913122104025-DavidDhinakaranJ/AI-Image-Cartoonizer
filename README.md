# AI Image Cartoonizer (One-Week Model-Improvement Demo)

Local full-stack project focused on **model output improvement** using an
Adaptive Quality Enhancement (AQE) pipeline.

## What is implemented

- FastAPI backend with required endpoints:
  - `GET /api/health`
  - `GET /api/styles`
  - `POST /api/cartoonize` (`style_id`, `variant=baseline|improved`)
- 3 styles with external config in `backend/style_presets.json`:
  - `hayao`, `shinkai`, `paprika`
- AQE improvement pipeline:
  - Preprocess: EXIF fix, RGB conversion, resize, LAB brightness normalization, bilateral denoise
  - Postprocess: edge reinforcement, color quantization, contrast/saturation harmonization, unsharp mask
- Minimal React frontend:
  - upload image
  - select style
  - generate and compare baseline vs improved outputs
  - view metrics and download improved output
- Evaluation tooling:
  - `scripts/evaluate_variants.py`
  - human-study template CSV in `evaluation/human_study_template.csv`

## Model notes

The service is ready for AnimeGAN-style ONNX checkpoints.

Place checkpoints here:

- `backend/models/hayao.onnx`
- `backend/models/shinkai.onnx`
- `backend/models/paprika.onnx`

If checkpoints are missing, the backend automatically uses a deterministic OpenCV
fallback stylization so the demo still runs end-to-end.

## Quick start (Windows / PowerShell)

1. Install prerequisites:
- Python 3.11+
- Node.js 18+

2. Setup:

```powershell
.\scripts\setup.ps1
```

3. Run backend + frontend:

```powershell
.\scripts\run-local.ps1
```

4. Open:
- Frontend: `http://localhost:5173`
- Backend docs: `http://localhost:8000/docs`

## API contract

### `GET /api/styles`

Returns:

```json
{
  "styles": [
    {"id":"hayao","name":"Hayao","preview":"/static/previews/hayao.svg"},
    {"id":"shinkai","name":"Shinkai","preview":"/static/previews/shinkai.svg"},
    {"id":"paprika","name":"Paprika","preview":"/static/previews/paprika.svg"}
  ]
}
```

### `POST /api/cartoonize`

Form-data:
- `image` (`jpg|jpeg|png|webp`)
- `style_id` (`hayao|shinkai|paprika`)
- `variant` (`baseline|improved`)

Response:

```json
{
  "result_url": "/api/results/xxx.png",
  "style_id": "hayao",
  "variant": "improved",
  "latency_ms": 820,
  "metrics": {
    "edge_ssim": 0.74,
    "artifact_score": 0.18
  }
}
```

## Style tuning

Adjust AQE behavior per style in `backend/style_presets.json`:

- `resize_max`
- `denoise_strength`
- `edge_weight`
- `color_quant_k`
- `contrast_gain`
- `sharpen_amount`
- `saturation_gain`

## Evaluation workflow

1. Run app backend (`http://localhost:8000`).
2. Put test images into a folder, e.g. `samples/`.
3. Execute:

```powershell
python .\scripts\evaluate_variants.py --input-dir .\samples --output-csv .\evaluation\results.csv
```

Generated CSV includes baseline/improved metrics and latency per image/style.

## Tests

From `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Project structure

- `backend/` FastAPI + AQE + model inference service
- `frontend/` React + Vite demo UI
- `scripts/` setup, run, and evaluation automation
- `evaluation/` report templates for user-study evidence
