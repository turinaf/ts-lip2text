# Web Demo (FastAPI)

This demo provides a browser UI for:
- generating a prompt (`digit` or `grid`)
- recording webcam video while speaking the prompt
- showing live lip landmarks during capture
- submitting video for sequence verification

## Run

From project root:

```bash
conda activate torch
pip install -r demo/requirements.txt
uvicorn demo.app:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

## Notes

- The app reuses existing project checkpoints:
  - `models/digit/best_sequence_verifier.pt`
  - `models/grid/best_sequence_verifier.pt`
- Landmark extraction in backend uses `data/face_landmarker.task`.
- Frontend landmark overlay uses MediaPipe Face Mesh in the browser.
