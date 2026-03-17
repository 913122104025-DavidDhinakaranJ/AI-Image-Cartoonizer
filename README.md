# AI Image Cartoonizer (One-Week Model-Improvement Demo)

Local full-stack project focused on **model output improvement** using an
Adaptive Quality Enhancement (AQE) pipeline.

## What is implemented

- FastAPI backend with required endpoints:
  - `GET /api/health`
  - `GET /api/styles`
  - `POST /api/cartoonize` (`style_id`, `variant=baseline|improved|improved_lite`)
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
  - `scripts/tune_style_presets.py`
  - `scripts/plot_results.py`
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
- `variant` (`baseline|improved|improved_lite`)

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

1. Build a fixed evaluation set (recommended `80-120` images):
- `samples/portrait/`
- `samples/landscape/`
- `samples/object/`

2. Run app backend (`http://localhost:8000`).

3. Run scoring:

```powershell
python .\scripts\evaluate_variants.py `
  --input-dir .\samples `
  --output-csv .\evaluation\results.csv `
  --summary-csv .\evaluation\summary.csv `
  --compare-variant improved_lite
```

Outputs:
- `evaluation/results.csv` (per-image, per-style metrics and deltas)
- `evaluation/summary.csv` (style-level means and win rates)

4. Optional: limit the run during iteration:

```powershell
python .\scripts\evaluate_variants.py --input-dir .\samples --max-images 30
```

## Auto-tune AQE presets

Searches parameter combinations and picks best per style against baseline.

Dry-run (does not modify config):

```powershell
python .\scripts\tune_style_presets.py `
  --input-dir .\samples `
  --styles hayao,shinkai,paprika `
  --max-images 24 `
  --max-trials 20 `
  --trials-csv .\evaluation\tuning_trials.csv `
  --best-json .\evaluation\tuned_presets.json
```

Apply tuned values directly to `backend/style_presets.json`:

```powershell
python .\scripts\tune_style_presets.py `
  --input-dir .\samples `
  --styles hayao,shinkai,paprika `
  --max-images 24 `
  --max-trials 20 `
  --apply
```

## Generate report charts

Create report-ready PNG charts from evaluation outputs:

```powershell
python .\scripts\plot_results.py `
  --summary-csv .\evaluation\summary.csv `
  --trials-csv .\evaluation\tuning_trials.csv `
  --output-dir .\evaluation\plots
```

Expected outputs:
- `evaluation/plots/quality_comparison.png`
- `evaluation/plots/latency_comparison.png`
- `evaluation/plots/win_rates.png`
- `evaluation/plots/tuning_hayao.png` (if trials exist)
- `evaluation/plots/tuning_shinkai.png` (if trials exist)
- `evaluation/plots/tuning_paprika.png` (if trials exist)

## Create Improved ONNX (Distillation)

This pipeline creates a true student ONNX model from your improved outputs.

1. Install distillation dependencies:

```powershell
# CPU/GPU generic install
.\backend\.venv\Scripts\python.exe -m pip install -r .\training\requirements.txt

# NVIDIA CUDA build (recommended for RTX GPU)
.\backend\.venv\Scripts\python.exe -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121

# Optional fallback when CUDA torch is unavailable on Windows
.\backend\.venv\Scripts\python.exe -m pip install torch-directml
```

2. Generate teacher-student pairs (run per style):

```powershell
.\backend\.venv\Scripts\python.exe .\scripts\generate_teacher_pairs.py `
  --input-dir .\samples `
  --style-id hayao `
  --variant improved_lite `
  --max-images 300
```

3. Train student model (run per style):

```powershell
.\backend\.venv\Scripts\python.exe .\scripts\train_student_distill.py `
  --manifest .\training\distill_data\hayao\manifest.csv `
  --style-id hayao `
  --epochs 8 `
  --batch-size 2 `
  --image-size 256
```

4. Export student to ONNX:

```powershell
.\backend\.venv\Scripts\python.exe .\scripts\export_student_onnx.py `
  --checkpoint .\training\artifacts\hayao_student_best.pt `
  --output .\backend\models\hayao_v2.onnx `
  --image-size 512
```

5. Repeat for `shinkai` and `paprika`, then update `backend/style_presets.json` model paths:
- `models/hayao_v2.onnx`
- `models/shinkai_v2.onnx`
- `models/paprika_v2.onnx`

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
