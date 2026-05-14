"""
Lip-Text Verification Inference
--------------------------------
Given a raw video and a claimed digit string, extract lip features
and verify whether the lip movements match the claimed digits.

Usage:
    python inference.py --video path/to/video.mp4 --digits "1 3 5 7 9 2 4 6"
    python inference.py --video path/to/video.mp4 --lab path/to/annotation.lab
    python inference.py --video path/to/video.mp4 --digits "1 3 5 7 9 2 4 6" --mode digit
    python inference.py --video path/to/video.mp4 --mode seq2seq --lab path/to/annotation.lab
"""
import cv2
import librosa
import mediapipe as mp_lib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import argparse
import os

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


def parse_lab_file(lab_path, fps):
    """Parse .lab annotation: line 1 = digits, line 2 = time ranges."""
    with open(lab_path) as f:
        lines = f.read().strip().split('\n')
    digits = lines[0].strip().split()
    time_ranges = lines[1].strip().split()
    alignments = []
    for digit, tr in zip(digits, time_ranges):
        ss, es = tr.split('-')
        sf = int(float(ss) * fps)
        ef = int(float(es) * fps)
        alignments.append((sf, ef, digit))
    return digits, alignments


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
def infer_sequence(model, segments, digits, device):
    """Run sequence-level verification."""
    all_feats, all_masks = [], []
    for seg in segments:
        f, m = pad_segment(seg, MAX_SEQ_LEN)
        all_feats.append(f)
        all_masks.append(m)

    feats_t = torch.FloatTensor(np.array([all_feats])).to(device)    # (1, 8, T, 5)
    masks_t = torch.FloatTensor(np.array([all_masks])).to(device)    # (1, 8, T)
    digits_idx = [CHAR_TO_IDX[d] for d in digits]
    digits_t = torch.LongTensor([digits_idx]).to(device)             # (1, 8)

    with torch.no_grad():
        logit = model(feats_t, masks_t, digits_t)
        prob = torch.sigmoid(logit).item()

    return prob


def infer_per_digit(model, segments, digits, device):
    """Run per-digit verification, return per-digit probabilities."""
    results = []
    for seg, digit in zip(segments, digits):
        feat, mask = pad_segment(seg, MAX_SEQ_LEN)
        feat_t = torch.FloatTensor(feat).unsqueeze(0).to(device)    # (1, T, 5)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)    # (1, T)
        digit_idx = CHAR_TO_IDX[digit]
        digit_t = torch.LongTensor([[digit_idx]]).to(device)        # (1, 1)

        with torch.no_grad():
            logit = model(feat_t, mask_t, digit_t)
            prob = torch.sigmoid(logit).item()

        results.append((digit, prob))
    return results


def infer_seq2seq(model, segments, device, max_len):
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
            bos_idx=BOS_IDX,
            max_len=max_len,
        )[0].cpu().tolist()

    out = []
    for tok in pred_tokens:
        if tok == EOS_IDX or tok == PAD_IDX:
            break
        out.append(tok)
    return out


# --- Main ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lip-text verification inference')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--digits', type=str,
                       help='Claimed digit string, space-separated (e.g. "1 3 5 7 9 2 4 6")')
    group.add_argument('--lab', type=str,
                       help='Path to .lab annotation file (contains digits + time ranges)')
    parser.add_argument('--mode', choices=['digit', 'sequence', 'seq2seq'], default='sequence',
                        help='Verification mode (default: sequence)')
    parser.add_argument('--n_digits', type=int, default=None,
                        help='Required for seq2seq mode when no .lab is provided')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--face_model', type=str, default=FACE_MODEL_PATH,
                        help=f'Path to face landmarker model (default: {FACE_MODEL_PATH})')
    args = parser.parse_args()

    default_model_name = {
        'digit': 'best_digit_verifier.pt',
        'sequence': 'best_sequence_verifier.pt',
        'seq2seq': 'best_seq2seq.pt',
    }[args.mode]
    model_path = args.model_path or os.path.join(MODEL_DIR, default_model_name)

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        exit(1)
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        exit(1)
    if not os.path.exists(args.face_model):
        print(f"ERROR: Face landmarker model not found: {args.face_model}")
        exit(1)

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
    rms = extract_rms_from_video(args.video, num_frames, fps)
    features = np.column_stack([features, rms])
    print(f"  Features shape: {features.shape}")

    # 3. Parse digits and segment
    if args.lab:
        digits, alignments = parse_lab_file(args.lab, fps)
        segments = segment_by_time(features, alignments, num_frames)
        print(f"  Digits from .lab: {' '.join(digits)}")
    else:
        if args.mode == 'seq2seq':
            if args.n_digits is None:
                print('ERROR: seq2seq inference needs --lab or --n_digits to define segmentation length')
                exit(1)
            n_digits = args.n_digits
            print(f"  No .lab provided — auto-segmenting by lip aperture minima...")
            segments = segment_by_aperture(features, n_digits, fps)
        else:
            digits = args.digits.strip().split()
            n_digits = len(digits)
            # Auto-segment using lip aperture minima (mouth closes between digits)
            print(f"  No .lab provided — auto-segmenting by lip aperture minima...")
            segments = segment_by_aperture(features, n_digits, fps)
            print(f"  Digits: {' '.join(digits)}")

    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    n_features = infer_input_feature_dim(state_dict)
    features = adapt_feature_dim(features, n_features)
    if args.lab:
        segments = [adapt_feature_dim(seg, n_features) for seg in segments]
    else:
        segments = [adapt_feature_dim(seg, n_features) for seg in segments]

    if digits is not None:
        # Validate digits when a claimed sequence is available.
        for d in digits:
            if d not in CHAR_TO_IDX:
                print(f"ERROR: Unknown digit '{d}'. Valid: {VOCAB}")
                exit(1)

    # 4. Load model
    if args.mode == 'digit':
        model = DigitVerifier(n_classes=N_CLASSES, embed_dim=EMBED_DIM,
                              n_features=n_features, hidden_dim=HIDDEN_DIM).to(DEVICE)
    elif args.mode == 'sequence':
        model = SequenceVerifier(n_classes=N_CLASSES, embed_dim=EMBED_DIM,
                                 n_features=n_features, hidden_dim=HIDDEN_DIM).to(DEVICE)
    else:
        model = TinyLipSeq2Seq(
            vocab_size=SEQ2SEQ_VOCAB_SIZE,
            pad_idx=PAD_IDX,
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
        ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded model: {model_path}")

    # 5. Run inference
    print(f"\n{'='*50}")
    print(f"  Mode: {args.mode}")
    print(f"{'='*50}")

    if args.mode == 'seq2seq':
        pred_ids = infer_seq2seq(model, segments, DEVICE, max_len=(n_digits + 1))
        pred_digits = [VOCAB[i] for i in pred_ids]
        print(f"\n  Predicted digits: {' '.join(pred_digits) if pred_digits else '(empty)'}")
        if args.lab:
            print(f"  Ground truth: {' '.join(digits)}")
    elif args.mode == 'sequence':
        prob = infer_sequence(model, segments, digits, DEVICE)
        print(f"\n  Claimed digits: {' '.join(digits)}")
        print(f"  Sequence probability: {prob:.4f}")
    else:
        digit_results = infer_per_digit(model, segments, digits, DEVICE)
        print(f"\n  Per-digit probabilities:")
        for digit, prob in digit_results:
            print(f"    Digit '{digit}': {prob:.4f}")
        avg_prob = np.mean([p for _, p in digit_results])
        print(f"\n  Average probability: {avg_prob:.4f}")

    print()
