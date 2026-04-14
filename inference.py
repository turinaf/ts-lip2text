"""
Lip-Text Verification Inference
--------------------------------
Given a raw video and a claimed digit string, extract lip features
and verify whether the lip movements match the claimed digits.

Usage:
    python inference.py --video path/to/video.mp4 --digits "1 3 5 7 9 2 4 6"
    python inference.py --video path/to/video.mp4 --lab path/to/annotation.lab
    python inference.py --video path/to/video.mp4 --digits "1 3 5 7 9 2 4 6" --mode digit
"""
import cv2
import mediapipe as mp_lib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import argparse
import os

from model import DigitVerifier, SequenceVerifier, CHAR_TO_IDX, VOCAB, N_CLASSES

# --- Configuration ---
FACE_MODEL_PATH = 'data/face_landmarker.task'
MODEL_DIR = 'models'
MAX_SEQ_LEN = 30
N_FEATURES = 5
EMBED_DIM = 64
HIDDEN_DIM = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')

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


def compute_features(all_lm):
    """Compute 5D lip features from face landmarks: (T, 5)."""
    num_frames = len(all_lm)
    interocular = np.linalg.norm(
        all_lm[:, LEFT_EYE_OUTER] - all_lm[:, RIGHT_EYE_OUTER], axis=1
    )
    interocular = np.where(interocular == 0, 1.0, interocular)

    vert_ap = np.linalg.norm(
        all_lm[:, IDX_TOP_INNER] - all_lm[:, IDX_BOTTOM_INNER], axis=1
    ) / interocular
    horiz_sp = np.linalg.norm(
        all_lm[:, IDX_LEFT_CORNER] - all_lm[:, IDX_RIGHT_CORNER], axis=1
    ) / interocular

    inner_areas = np.zeros(num_frames)
    compactness = np.zeros(num_frames)
    for i in range(num_frames):
        lm = all_lm[i]
        ia = polygon_area(lm[INNER_LIP])
        oa = polygon_area(lm[OUTER_LIP])
        op = polygon_perimeter(lm[OUTER_LIP])
        inner_areas[i] = ia / (interocular[i] ** 2)
        compactness[i] = (4 * np.pi * oa) / (op ** 2 + 1e-8)

    vv = np.gradient(vert_ap)
    hv = np.gradient(horiz_sp)
    lip_speed = np.sqrt(vv ** 2 + hv ** 2)

    return np.column_stack([vert_ap, horiz_sp, inner_areas, compactness, lip_speed])


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


def pad_segment(seg, max_len):
    """Pad/truncate a segment to max_len, return (feat, mask)."""
    T = seg.shape[0]
    if T >= max_len:
        feat = seg[:max_len].astype(np.float32)
        mask = np.ones(max_len, dtype=np.float32)
    else:
        feat = np.zeros((max_len, N_FEATURES), dtype=np.float32)
        feat[:T] = seg
        mask = np.zeros(max_len, dtype=np.float32)
        mask[:T] = 1.0
    return feat, mask


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
    parser.add_argument('--mode', choices=['digit', 'sequence'], default='sequence',
                        help='Verification mode (default: sequence)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold (default: 0.5)')
    parser.add_argument('--face_model', type=str, default=FACE_MODEL_PATH,
                        help=f'Path to face landmarker model (default: {FACE_MODEL_PATH})')
    args = parser.parse_args()

    model_path = args.model_path or os.path.join(MODEL_DIR, f'best_{args.mode}_verifier.pt')

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
    print(f"  Features shape: {features.shape}")

    # 3. Parse digits and segment
    if args.lab:
        digits, alignments = parse_lab_file(args.lab, fps)
        segments = segment_by_time(features, alignments, num_frames)
        print(f"  Digits from .lab: {' '.join(digits)}")
    else:
        digits = args.digits.strip().split()
        # Without .lab, split video evenly across digits
        n_digits = len(digits)
        frames_per_digit = num_frames // n_digits
        segments = []
        for i in range(n_digits):
            sf = i * frames_per_digit
            ef = (i + 1) * frames_per_digit if i < n_digits - 1 else num_frames
            segments.append(features[sf:ef])
        print(f"  Digits (even split): {' '.join(digits)} ({frames_per_digit} frames each)")

    # Validate digits
    for d in digits:
        if d not in CHAR_TO_IDX:
            print(f"ERROR: Unknown digit '{d}'. Valid: {VOCAB}")
            exit(1)

    # 4. Load model
    if args.mode == 'digit':
        model = DigitVerifier(n_classes=N_CLASSES, embed_dim=EMBED_DIM,
                              n_features=N_FEATURES, hidden_dim=HIDDEN_DIM).to(DEVICE)
    else:
        model = SequenceVerifier(n_classes=N_CLASSES, embed_dim=EMBED_DIM,
                                 n_features=N_FEATURES, hidden_dim=HIDDEN_DIM).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"  Loaded model: {model_path}")

    # 5. Run inference
    print(f"\n{'='*50}")
    print(f"  Claimed digits: {' '.join(digits)}")
    print(f"  Mode: {args.mode}, Threshold: {args.threshold}")
    print(f"{'='*50}")

    if args.mode == 'sequence':
        prob = infer_sequence(model, segments, digits, DEVICE)
        verdict = "MATCH" if prob >= args.threshold else "MISMATCH"
        print(f"\n  Sequence probability: {prob:.4f}")
        print(f"  Verdict: {verdict}")
    else:
        digit_results = infer_per_digit(model, segments, digits, DEVICE)
        print(f"\n  Per-digit results:")
        all_match = True
        for digit, prob in digit_results:
            verdict = "MATCH" if prob >= args.threshold else "MISMATCH"
            if prob < args.threshold:
                all_match = False
            print(f"    Digit '{digit}': {prob:.4f} -> {verdict}")
        overall = "MATCH" if all_match else "MISMATCH"
        avg_prob = np.mean([p for _, p in digit_results])
        print(f"\n  Average probability: {avg_prob:.4f}")
        print(f"  Overall verdict: {overall}")

    print()
