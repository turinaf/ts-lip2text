"""
Lip-Text Verification Inference
--------------------------------
Given a raw video and a claimed token sequence, extract lip features
and verify whether the lip movements match the claim.

Usage:
    python inference.py --video path/to/video.mp4 --digits "1 3 5 7 9 2 4 6"
    python inference.py --video path/to/video.mp4 --dataset grid --digits "place blue at a five please"
    python inference.py --video path/to/video.mp4 --lab path/to/annotation.lab
    python inference.py --video path/to/video.mp4 --digits "1 3 5 7 9 2 4 6" --mode digit
    python inference.py --video path/to/video.mp4 --mode seq2seq --lab path/to/annotation.lab
"""
import cv2
import mediapipe as mp_lib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import argparse
import os
import json

from model import (DigitVerifier, SequenceVerifier,
                   TinyLipSeq2Seq, CHAR_TO_IDX, VOCAB, N_CLASSES)

# --- Configuration ---
FACE_MODEL_PATH = 'data/face_landmarker.task'
MODEL_DIR = 'models'
MAX_SEQ_LEN = 30
EMBED_DIM = 64
HIDDEN_DIM = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
PAD_IDX = N_CLASSES
BOS_IDX = N_CLASSES + 1
EOS_IDX = N_CLASSES + 2
SEQ2SEQ_VOCAB_SIZE = N_CLASSES + 3


def _dataset_model_dir(dataset, encoder_type):
    if encoder_type == 'transformer':
        return os.path.join(MODEL_DIR, 'transformer_encoder')
    if dataset == 'digit':
        return MODEL_DIR
    return os.path.join(MODEL_DIR, dataset)


def _default_model_path(dataset, mode, encoder_type):
    model_dir = _dataset_model_dir(dataset, encoder_type)
    if mode == 'seq2seq':
        return os.path.join(model_dir, 'best_seq2seq.pt')
    return os.path.join(model_dir, f'best_{mode}_verifier.pt')


def _default_vocab_path(dataset, encoder_type):
    return os.path.join(_dataset_model_dir(dataset, encoder_type), 'vocab.json')


def _load_token_mappings(dataset, vocab_path):
    if dataset == 'digit' and vocab_path is None:
        token_to_idx = dict(CHAR_TO_IDX)
    else:
        if vocab_path is None:
            raise RuntimeError('Vocabulary path is required for non-digit datasets')
        with open(vocab_path) as f:
            token_to_idx = json.load(f)
        token_to_idx = {str(token): int(idx) for token, idx in token_to_idx.items()}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return token_to_idx, idx_to_token


def _detect_encoder_type(state_dict):
    if any(key.startswith('lip_encoder.gru') for key in state_dict):
        return 'bigru'
    if any(key.startswith('seg_encoder.gru') for key in state_dict):
        return 'bigru'
    return 'transformer'

# Lip landmark indices
OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
             185, 40, 39, 37, 0, 267, 269, 270, 409]
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
             191, 80, 81, 82, 13, 312, 311, 310, 415]
IDX_TOP_INNER = 13
IDX_BOTTOM_INNER = 14
IDX_LEFT_CORNER = 61
IDX_RIGHT_CORNER = 291
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


# --- Feature extraction (same as preprocess.py) ---
def polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_perimeter(points):
    diffs = np.diff(points, axis=0, append=points[:1])
    return np.sum(np.linalg.norm(diffs, axis=1))


def extract_landmarks(video_path, detector):
    """Extract 478 face landmarks per frame from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmarks_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if result.face_landmarks:
            face_lm = result.face_landmarks[0]
            lm = np.array([[l.x * w, l.y * h] for l in face_lm])
            landmarks_list.append(lm)
        else:
            if landmarks_list:
                landmarks_list.append(landmarks_list[-1])
            else:
                landmarks_list.append(np.zeros((478, 2)))
    cap.release()

    if not landmarks_list:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return np.array(landmarks_list), fps


def extract_rms_from_video(video_path, num_frames, fps):
    """Extract per-video-frame RMS energy from the audio track of an mp4."""
    try:
        import librosa

        y, sr = librosa.load(video_path, sr=None, mono=True)
        hop_length = max(1, int(sr / fps))
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        audio_times = np.arange(len(rms)) * hop_length / sr
        video_times = np.arange(num_frames) / fps
        return np.interp(video_times, audio_times, rms)
    except Exception:
        return np.zeros(num_frames, dtype=np.float32)


def compute_features(all_lm):
    """Compute lip features from face landmarks: (T, 7)."""
    num_frames = len(all_lm)
    interocular = np.linalg.norm(
        all_lm[:, LEFT_EYE_OUTER] - all_lm[:, RIGHT_EYE_OUTER], axis=1
    )
    interocular = np.where(interocular == 0, 1.0, interocular)

    vert_ap = np.linalg.norm(
        all_lm[:, IDX_TOP_INNER] - all_lm[:, IDX_BOTTOM_INNER], axis=1
    ) / interocular
    outer_vert_ap = np.linalg.norm(
        all_lm[:, 17] - all_lm[:, 0], axis=1
    ) / interocular
    horiz_sp = np.linalg.norm(
        all_lm[:, IDX_LEFT_CORNER] - all_lm[:, IDX_RIGHT_CORNER], axis=1
    ) / interocular

    inner_areas = np.zeros(num_frames)
    outer_areas = np.zeros(num_frames)
    compactness = np.zeros(num_frames)
    for i in range(num_frames):
        lm = all_lm[i]
        ia = polygon_area(lm[INNER_LIP])
        oa = polygon_area(lm[OUTER_LIP])
        op = polygon_perimeter(lm[OUTER_LIP])
        inner_areas[i] = ia / (interocular[i] ** 2)
        outer_areas[i] = oa / (interocular[i] ** 2)
        compactness[i] = (4 * np.pi * oa) / (op ** 2 + 1e-8)

    vv = np.gradient(vert_ap)
    hv = np.gradient(horiz_sp)
    lip_speed = np.sqrt(vv ** 2 + hv ** 2)

    return np.column_stack([
        vert_ap,
        outer_vert_ap,
        horiz_sp,
        inner_areas,
        outer_areas,
        compactness,
        lip_speed,
    ])


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def parse_lab_file(lab_path, fps, dataset='digit', num_frames=None):
    """Parse digit .lab or GRID .align annotation into frame-level alignments."""
    with open(lab_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        return [], []

    if dataset == 'grid':
        raw_entries = []
        max_end = 0.0
        for ln in lines:
            parts = ln.split()
            if len(parts) < 3 or not _is_number(parts[0]) or not _is_number(parts[1]):
                continue
            st = float(parts[0])
            en = float(parts[1])
            token = parts[2]
            raw_entries.append((st, en, token))
            max_end = max(max_end, en)

        if num_frames is not None and fps is not None and max_end > 0:
            duration_s = num_frames / fps
            sec_per_unit = duration_s / max_end
        else:
            # GRID/HTK default time unit: 1/25000 second.
            sec_per_unit = 1.0 / 25000.0

        alignments = []
        for st, en, token in raw_entries:
            if token.lower() in {'sil', 'sp'}:
                continue
            sf = int(round(st * sec_per_unit * fps))
            ef = int(round(en * sec_per_unit * fps))
            alignments.append((sf, ef, token))

        tokens = [tok for _, _, tok in alignments]
        return tokens, alignments

    if len(lines) < 2:
        return [], []

    tokens = lines[0].split()
    time_ranges = lines[1].split()
    alignments = []
    for token, tr in zip(tokens, time_ranges):
        ss, es = tr.split('-')
        sf = int(float(ss) * fps)
        ef = int(float(es) * fps)
        alignments.append((sf, ef, token))
    return tokens, alignments


def segment_by_time(features, alignments, num_frames):
    """Split features into per-digit segments using alignments."""
    segments = []
    for sf, ef, digit in alignments:
        sf = max(0, min(sf, num_frames - 1))
        ef = max(sf + 1, min(ef, num_frames))
        segments.append(features[sf:ef])
    return segments


def segment_by_aperture(features, n_digits, fps, smooth_window=3):
    """
    Automatically segment a video into n_digits intervals by detecting
    local minima in vertical aperture (feature 0) — the mouth closes
    between digits. Picks the n_digits-1 deepest minima as boundaries.
    """
    from scipy.signal import savgol_filter, argrelmin

    aperture = features[:, 0]
    T = len(aperture)

    # Smooth to suppress within-digit noise (window must be odd)
    win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    win = min(win, T if T % 2 == 1 else T - 1)
    smoothed = savgol_filter(aperture, window_length=win, polyorder=1) if T > win else aperture

    # Find all local minima at order=1 (most sensitive), then pick deepest n_digits-1
    minima_idx = argrelmin(smoothed, order=1)[0]

    if len(minima_idx) >= n_digits - 1:
        depths = smoothed[minima_idx]
        chosen = minima_idx[np.argsort(depths)[:n_digits - 1]]
    else:
        # Fewer minima than needed: supplement with the lowest points in
        # equally-spaced windows to guarantee n_digits-1 boundaries
        step = T / n_digits
        window_mins = []
        for i in range(n_digits - 1):
            center = int(round((i + 1) * step))
            lo = max(0, center - int(step // 2))
            hi = min(T, center + int(step // 2))
            window_mins.append(lo + np.argmin(smoothed[lo:hi]))
        chosen = np.array(window_mins)

    boundaries = sorted(set(int(b) for b in chosen))

    # Build segments
    starts = [0] + boundaries
    ends = boundaries + [T]
    segments = [features[s:e] for s, e in zip(starts, ends)]

    frames_info = ' | '.join(f'{e-s}fr' for s, e in zip(starts, ends))
    print(f"  Auto-segmented: [{frames_info}]")
    return segments

    # Build segments from boundaries
    starts = [0] + boundaries
    ends = boundaries + [T]
    segments = [features[s:e] for s, e in zip(starts, ends)]

    frames_info = ' | '.join(f'{e-s}fr' for s, e in zip(starts, ends))
    print(f"  Auto-segmented: [{frames_info}]")
    return segments


def pad_segment(seg, max_len):
    """Pad/truncate a segment to max_len, return (feat, mask)."""
    T = seg.shape[0]
    if T >= max_len:
        feat = seg[:max_len].astype(np.float32)
        mask = np.ones(max_len, dtype=np.float32)
    else:
        feat = np.zeros((max_len, seg.shape[1]), dtype=np.float32)
        feat[:T] = seg
        mask = np.zeros(max_len, dtype=np.float32)
        mask[:T] = 1.0
    return feat, mask


def adapt_feature_dim(features, target_dim):
    """Slice or zero-pad feature columns to match the model input width."""
    current_dim = features.shape[1]
    if current_dim == target_dim:
        return features
    if current_dim > target_dim:
        return features[:, :target_dim]

    padded = np.zeros((features.shape[0], target_dim), dtype=np.float32)
    padded[:, :current_dim] = features
    return padded


def infer_input_feature_dim(state_dict):
    for key in (
        'seg_encoder.conv.0.weight',
        'lip_encoder.conv.0.weight',
    ):
        if key in state_dict:
            return state_dict[key].shape[1]
    raise RuntimeError('Could not infer feature dimension from checkpoint')


def infer_n_digits_from_segments(segments):
    return len(segments)


# --- Inference ---
def infer_sequence(model, segments, tokens, token_to_idx, device):
    """Run sequence-level verification."""
    all_feats, all_masks = [], []
    for seg in segments:
        f, m = pad_segment(seg, MAX_SEQ_LEN)
        all_feats.append(f)
        all_masks.append(m)

    feats_t = torch.FloatTensor(np.array([all_feats])).to(device)    # (1, 8, T, 5)
    masks_t = torch.FloatTensor(np.array([all_masks])).to(device)    # (1, 8, T)
    token_ids = [token_to_idx[token] for token in tokens]
    digits_t = torch.LongTensor([token_ids]).to(device)              # (1, S)

    with torch.no_grad():
        logit = model(feats_t, masks_t, digits_t)
        prob = torch.sigmoid(logit).item()

    return prob


def infer_per_digit(model, segments, tokens, token_to_idx, device):
    """Run per-digit verification, return per-digit probabilities."""
    results = []
    for seg, token in zip(segments, tokens):
        feat, mask = pad_segment(seg, MAX_SEQ_LEN)
        feat_t = torch.FloatTensor(feat).unsqueeze(0).to(device)    # (1, T, 5)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)    # (1, T)
        digit_idx = token_to_idx[token]
        digit_t = torch.LongTensor([[digit_idx]]).to(device)        # (1, 1)

        with torch.no_grad():
            logit = model(feat_t, mask_t, digit_t)
            prob = torch.sigmoid(logit).item()

        results.append((token, prob))
    return results


def infer_seq2seq(model, segments, device, max_len, bos_idx, pad_idx, eos_idx):
    """Run seq2seq transcription and return predicted token ids."""
    all_feats, all_masks = [], []
    for seg in segments:
        f, m = pad_segment(seg, MAX_SEQ_LEN)
        all_feats.append(f)
        all_masks.append(m)

    feats_t = torch.FloatTensor(np.array([all_feats])).to(device)
    masks_t = torch.FloatTensor(np.array([all_masks])).to(device)
    src_pad = torch.zeros((1, len(segments)), dtype=torch.bool, device=device)

    with torch.no_grad():
        pred_tokens = model.greedy_decode(
            feats_t,
            masks_t,
            src_pad,
            bos_idx=bos_idx,
            max_len=max_len,
        )[0].cpu().tolist()

    out = []
    for tok in pred_tokens:
        if tok == eos_idx or tok == pad_idx:
            break
        out.append(tok)
    return out


# --- Main ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lip-text verification inference')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--digits', type=str,
                        help='Claimed token string, space-separated (digits or GRID words)')
    parser.add_argument('--lab', type=str,
                        help='Path to .lab annotation file (contains tokens + time ranges)')
    parser.add_argument('--dataset', choices=['digit', 'grid'], default='digit',
                        help='Dataset-specific vocab/model layout (default: digit)')
    parser.add_argument('--mode', choices=['digit', 'sequence', 'seq2seq'], default='sequence',
                        help='Verification mode (default: sequence)')
    parser.add_argument('--n_digits', type=int, default=None,
                        help='Required for seq2seq mode when no .lab is provided')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab_path', type=str, default=None,
                        help='Path to vocabulary JSON (defaults to dataset-specific vocab)')
    parser.add_argument('--encoder', choices=['auto', 'bigru', 'transformer'], default='auto',
                        help='Encoder variant used by the checkpoint (default: auto-detect)')
    parser.add_argument('--face_model', type=str, default=FACE_MODEL_PATH,
                        help=f'Path to face landmarker model (default: {FACE_MODEL_PATH})')
    parser.add_argument(
        '--use-audio-rms',
        action='store_true',
        help='Append rms_energy from audio to frame features before inference.',
    )
    args = parser.parse_args()

    if args.dataset == 'grid' and args.mode == 'seq2seq' and args.model_path is None:
        default_grid_seq2seq = _default_model_path(args.dataset, args.mode, 'bigru')
        default_grid_transformer_seq2seq = _default_model_path(args.dataset, args.mode, 'transformer')
        if not os.path.exists(default_grid_seq2seq) and not os.path.exists(default_grid_transformer_seq2seq):
            print('ERROR: No default GRID seq2seq checkpoint found. Provide --model_path explicitly.')
            exit(1)

    if args.mode == 'seq2seq':
        if args.lab is None and args.n_digits is None:
            print('ERROR: seq2seq inference needs --lab or --n_digits to define segmentation length')
            exit(1)
    else:
        if args.lab is None and args.digits is None:
            print('ERROR: verification modes need --digits or --lab')
            exit(1)
        if args.lab is not None and args.digits is not None:
            print('ERROR: provide only one of --digits or --lab')
            exit(1)

    resolved_encoder = args.encoder
    if args.model_path:
        model_path = args.model_path
    elif args.encoder != 'auto':
        model_path = _default_model_path(args.dataset, args.mode, args.encoder)
    else:
        candidate_paths = []
        if args.dataset == 'digit':
            candidate_paths.extend([
                _default_model_path(args.dataset, args.mode, 'transformer'),
                _default_model_path(args.dataset, args.mode, 'bigru'),
            ])
        else:
            candidate_paths.extend([
                _default_model_path(args.dataset, args.mode, 'bigru'),
                _default_model_path(args.dataset, args.mode, 'transformer'),
            ])
        model_path = next((path for path in candidate_paths if os.path.exists(path)), candidate_paths[0])

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        exit(1)
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        exit(1)
    if not os.path.exists(args.face_model):
        print(f"ERROR: Face landmarker model not found: {args.face_model}")
        exit(1)

    vocab_path = args.vocab_path
    if vocab_path is None and args.dataset != 'digit':
        candidate_vocab_paths = [
            _default_vocab_path(args.dataset, resolved_encoder if resolved_encoder != 'auto' else 'bigru'),
            _default_vocab_path(args.dataset, 'transformer'),
        ]
        vocab_path = next((path for path in candidate_vocab_paths if os.path.exists(path)), None)

    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if resolved_encoder == 'auto':
        resolved_encoder = _detect_encoder_type(state_dict)
    if vocab_path is None and args.dataset != 'digit':
        vocab_path = _default_vocab_path(args.dataset, resolved_encoder)
    if vocab_path is not None and not os.path.exists(vocab_path):
        print(f"ERROR: Vocabulary not found: {vocab_path}")
        exit(1)

    token_to_idx, idx_to_token = _load_token_mappings(args.dataset, vocab_path)
    n_classes = len(token_to_idx)
    pad_idx = n_classes
    bos_idx = n_classes + 1
    eos_idx = n_classes + 2
    seq2seq_vocab_size = n_classes + 3

    # 1. Initialize face landmarker
    print(f"Device: {DEVICE}")
    base_options = python.BaseOptions(model_asset_path=args.face_model)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # 2. Extract landmarks and features
    print(f"Extracting landmarks from {args.video}...")
    all_lm, fps = extract_landmarks(args.video, detector)
    num_frames = len(all_lm)
    print(f"  {num_frames} frames @ {fps:.1f} FPS")

    features = compute_features(all_lm)
    if args.use_audio_rms:
        rms = extract_rms_from_video(args.video, num_frames, fps)
        features = np.column_stack([features, rms])
    print(f"  Features shape: {features.shape}")

    # 3. Parse tokens and segment
    tokens = None
    if args.lab:
        tokens, alignments = parse_lab_file(args.lab, fps, dataset=args.dataset, num_frames=num_frames)
        segments = segment_by_time(features, alignments, num_frames)
        print(f"  Tokens from .lab: {' '.join(tokens)}")
    else:
        if args.mode == 'seq2seq':
            if args.n_digits is None:
                print('ERROR: seq2seq inference needs --lab or --n_digits to define segmentation length')
                exit(1)
            n_digits = args.n_digits
            print(f"  No .lab provided — auto-segmenting by lip aperture minima...")
            segments = segment_by_aperture(features, n_digits, fps)
        else:
            tokens = args.digits.strip().split()
            n_digits = len(tokens)
            print(f"  No .lab provided — auto-segmenting by lip aperture minima...")
            segments = segment_by_aperture(features, n_digits, fps)
            print(f"  Tokens: {' '.join(tokens)}")

    n_features = infer_input_feature_dim(state_dict)
    features = adapt_feature_dim(features, n_features)
    segments = [adapt_feature_dim(seg, n_features) for seg in segments]
    n_digits = infer_n_digits_from_segments(segments)

    if tokens is not None:
        for token in tokens:
            if token not in token_to_idx:
                valid_tokens = ', '.join(sorted(token_to_idx.keys()))
                print(f"ERROR: Unknown token '{token}'. Valid: {valid_tokens}")
                exit(1)

    # 4. Load model
    if args.mode == 'digit':
        model = DigitVerifier(n_classes=n_classes, embed_dim=EMBED_DIM,
                              n_features=n_features, hidden_dim=HIDDEN_DIM,
                              encoder_type=resolved_encoder).to(DEVICE)
    elif args.mode == 'sequence':
        model = SequenceVerifier(n_classes=n_classes, embed_dim=EMBED_DIM,
                                 n_features=n_features, hidden_dim=HIDDEN_DIM,
                                 encoder_type=resolved_encoder).to(DEVICE)
    else:
        model = TinyLipSeq2Seq(
            vocab_size=seq2seq_vocab_size,
            pad_idx=pad_idx,
            n_features=n_features,
            seg_embed_dim=48,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            ff_dim=128,
            dropout=0.1,
            max_src_len=12,
            max_tgt_len=12,
            hidden_dim=64,
            encoder_type=resolved_encoder,
        ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Dataset: {args.dataset}")
    print(f"  Encoder: {resolved_encoder}")
    print(f"  Loaded model: {model_path}")
    if vocab_path:
        print(f"  Loaded vocab: {vocab_path}")

    # 5. Run inference
    print(f"\n{'='*50}")
    print(f"  Mode: {args.mode}")
    print(f"{'='*50}")

    if args.mode == 'seq2seq':
        pred_ids = infer_seq2seq(
            model,
            segments,
            DEVICE,
            max_len=(n_digits + 1),
            bos_idx=bos_idx,
            pad_idx=pad_idx,
            eos_idx=eos_idx,
        )
        pred_tokens = [idx_to_token[i] for i in pred_ids if i in idx_to_token]
        print(f"\n  Predicted tokens: {' '.join(pred_tokens) if pred_tokens else '(empty)'}")
        if args.lab:
            print(f"  Ground truth: {' '.join(tokens)}")
    elif args.mode == 'sequence':
        prob = infer_sequence(model, segments, tokens, token_to_idx, DEVICE)
        print(f"\n  Claimed tokens: {' '.join(tokens)}")
        print(f"  Sequence probability: {prob:.4f}")
    else:
        digit_results = infer_per_digit(model, segments, tokens, token_to_idx, DEVICE)
        print(f"\n  Per-token probabilities:")
        for token, prob in digit_results:
            print(f"    Token '{token}': {prob:.4f}")
        avg_prob = np.mean([p for _, p in digit_results])
        print(f"\n  Average probability: {avg_prob:.4f}")

    print()
