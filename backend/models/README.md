# Model Files

Place your style-specific ONNX checkpoints in this directory:

- `hayao.onnx`
- `shinkai.onnx`
- `paprika.onnx`

The backend reads these paths from `backend/style_presets.json`.

If files are absent, the app runs with the built-in fallback stylization so you
can still demo baseline vs improved AQE behavior.
