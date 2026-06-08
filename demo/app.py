import json
import random
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Allow imports from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import (  # noqa: E402
    adapt_feature_dim,
    compute_features,
    extract_landmarks,
    infer_input_feature_dim,
    pad_segment,
    segment_by_aperture,
)
from model import SequenceVerifier  # noqa: E402


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
MAX_SEQ_LEN = 30
EMBED_DIM = 64
HIDDEN_DIM = 128

DATASET_CONFIG = {
    "digit": {
        "model_path": PROJECT_ROOT / "models" / "digit" / "best_sequence_verifier.pt",
        "vocab_path": PROJECT_ROOT / "models" / "digit" / "vocab.json",
        "results_path": PROJECT_ROOT / "models" / "digit" / "results_sequence.json",
        "expected_len": 8,
    },
    "grid": {
        "model_path": PROJECT_ROOT / "models" / "grid" / "best_sequence_verifier.pt",
        "vocab_path": PROJECT_ROOT / "models" / "grid" / "vocab.json",
        "results_path": PROJECT_ROOT / "models" / "grid" / "results_sequence.json",
        "expected_len": None,
    },
}

FACE_MODEL_PATH = PROJECT_ROOT / "data" / "face_landmarker.task"


class ModelBundle:
    def __init__(self, model, token_to_idx: Dict[str, int], threshold: float, n_features: int):
        self.model = model
        self.token_to_idx = token_to_idx
        self.threshold = threshold
        self.n_features = n_features


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _ensure_paths_exist()
    _get_detector()
    try:
        yield
    finally:
        global _detector
        if _detector is not None and hasattr(_detector, "close"):
            _detector.close()
            _detector = None


app = FastAPI(title="TS Lip2Text Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_detector = None
_model_cache: Dict[str, ModelBundle] = {}


def _ensure_paths_exist():
    missing = []
    if not FACE_MODEL_PATH.exists():
        missing.append(str(FACE_MODEL_PATH))
    for ds, cfg in DATASET_CONFIG.items():
        for key in ("model_path", "vocab_path", "results_path"):
            path = cfg[key]
            if not path.exists():
                missing.append(f"[{ds}] {path}")
    if missing:
        joined = "\n".join(missing)
        raise RuntimeError(f"Missing required files:\n{joined}")


def _get_detector():
    global _detector
    if _detector is not None:
        return _detector

    base_options = python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    _detector = vision.FaceLandmarker.create_from_options(options)
    return _detector


def _detect_encoder_type(state_dict: Dict[str, torch.Tensor]) -> str:
    if any(key.startswith("lip_encoder.gru") for key in state_dict.keys()):
        return "bigru"
    return "transformer"


def _load_threshold(results_path: Path) -> float:
    try:
        with results_path.open() as f:
            payload = json.load(f)
        return float(payload.get("final_metrics", {}).get("eer_threshold", 0.5))
    except Exception:
        return 0.5


def _load_bundle(dataset: str) -> ModelBundle:
    if dataset in _model_cache:
        return _model_cache[dataset]

    cfg = DATASET_CONFIG[dataset]
    with cfg["vocab_path"].open() as f:
        token_to_idx = json.load(f)

    state_dict = torch.load(cfg["model_path"], map_location=DEVICE, weights_only=True)
    n_features = int(infer_input_feature_dim(state_dict))
    encoder_type = _detect_encoder_type(state_dict)

    model = SequenceVerifier(
        n_classes=len(token_to_idx),
        embed_dim=EMBED_DIM,
        n_features=n_features,
        hidden_dim=HIDDEN_DIM,
        encoder_type=encoder_type,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    threshold = _load_threshold(cfg["results_path"])
    bundle = ModelBundle(model=model, token_to_idx=token_to_idx, threshold=threshold, n_features=n_features)
    _model_cache[dataset] = bundle
    return bundle


def _digit_prompt() -> List[str]:
    return [str(random.randint(0, 9)) for _ in range(8)]


def _grid_prompt(token_to_idx: Dict[str, int]) -> List[str]:
    vocab = set(token_to_idx.keys())

    commands = [tok for tok in ["bin", "lay", "place", "set"] if tok in vocab]
    colors = [tok for tok in ["blue", "green", "red", "white"] if tok in vocab]
    preps = [tok for tok in ["at", "by", "in", "with"] if tok in vocab]
    letters = [tok for tok in list("abcdefghijklmnopqrstuvwxyz") if tok in vocab]
    digits = [
        tok
        for tok in ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        if tok in vocab
    ]
    adverbs = [tok for tok in ["again", "now", "please", "soon"] if tok in vocab]

    if all([commands, colors, preps, letters, digits, adverbs]):
        return [
            random.choice(commands),
            random.choice(colors),
            random.choice(preps),
            random.choice(letters),
            random.choice(digits),
            random.choice(adverbs),
        ]

    fallback = sorted([t for t in vocab if t.isalpha() and len(t) > 0])
    if len(fallback) < 4:
        raise RuntimeError("GRID vocabulary is too small to build a prompt.")
    phrase_len = random.randint(4, min(6, len(fallback)))
    return random.sample(fallback, k=phrase_len)


def _normalize_tokens(dataset: str, prompt_text: str, token_to_idx: Dict[str, int]) -> List[str]:
    raw = prompt_text.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Prompt text cannot be empty.")

    if dataset == "digit":
        compact = raw.replace(" ", "")
        if not compact.isdigit() or len(compact) != 8:
            raise HTTPException(status_code=400, detail="Digit mode expects exactly 8 digits.")
        tokens = list(compact)
    else:
        tokens = [t.strip().lower() for t in raw.split() if t.strip()]
        if len(tokens) < 2:
            raise HTTPException(status_code=400, detail="GRID prompt should contain at least 2 words.")

    unknown = [tok for tok in tokens if tok not in token_to_idx]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown token(s) for {dataset} model: {', '.join(unknown)}",
        )

    return tokens


def _infer_sequence_probability(model, segments, tokens: List[str], token_to_idx: Dict[str, int]) -> float:
    all_feats = []
    all_masks = []
    for seg in segments:
        feat, mask = pad_segment(seg, MAX_SEQ_LEN)
        all_feats.append(feat)
        all_masks.append(mask)

    feats_t = torch.FloatTensor([all_feats]).to(DEVICE)
    masks_t = torch.FloatTensor([all_masks]).to(DEVICE)
    token_ids = torch.LongTensor([[token_to_idx[t] for t in tokens]]).to(DEVICE)

    with torch.no_grad():
        logit = model(feats_t, masks_t, token_ids)
        return float(torch.sigmoid(logit).item())


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/prompt")
def api_prompt(dataset: str):
    if dataset not in DATASET_CONFIG:
        raise HTTPException(status_code=400, detail="dataset must be 'digit' or 'grid'")

    bundle = _load_bundle(dataset)
    tokens = _digit_prompt() if dataset == "digit" else _grid_prompt(bundle.token_to_idx)
    return JSONResponse({"dataset": dataset, "tokens": tokens, "prompt": " ".join(tokens)})


@app.post("/api/verify")
def api_verify(
    dataset: str = Form(...),
    prompt_text: str = Form(...),
    video: UploadFile = File(...),
):
    if dataset not in DATASET_CONFIG:
        raise HTTPException(status_code=400, detail="dataset must be 'digit' or 'grid'")

    bundle = _load_bundle(dataset)
    tokens = _normalize_tokens(dataset, prompt_text, bundle.token_to_idx)

    suffix = Path(video.filename or "capture.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
        shutil.copyfileobj(video.file, tmp)

    try:
        detector = _get_detector()
        landmarks, fps = extract_landmarks(str(temp_path), detector)
        if landmarks is None or len(landmarks) == 0:
            raise HTTPException(status_code=400, detail="Could not read frames from uploaded video.")

        features = compute_features(landmarks).astype("float32")
        features = adapt_feature_dim(features, bundle.n_features).astype("float32")

        segments = segment_by_aperture(features, n_digits=len(tokens), fps=fps)
        if len(segments) != len(tokens):
            raise HTTPException(
                status_code=400,
                detail="Unable to segment the utterance to match the prompted token count.",
            )

        segments = [adapt_feature_dim(seg, bundle.n_features).astype("float32") for seg in segments]

        prob = _infer_sequence_probability(bundle.model, segments, tokens, bundle.token_to_idx)
        accepted = prob >= bundle.threshold

        return JSONResponse(
            {
                "dataset": dataset,
                "tokens": tokens,
                "probability": prob,
                "threshold": bundle.threshold,
                "accepted": accepted,
                "frames": int(len(landmarks)),
                "fps": float(fps),
            }
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": str(DEVICE)}
