# PitchCheck Rust Core

Optional PyO3 accelerator for TRIBE prediction post-processing.

Build into the active Python environment:

```bash
python -m pip install "maturin>=1.7,<2"
cd tribe_service/rust_core
python -m maturin develop --release --features extension-module
```

The Python service falls back to NumPy if `_pitchcheck_core` is not installed.
Set `PITCHCHECK_RUST_NUMERIC=0` to force the NumPy post-processing path for
comparison or emergency rollback.
