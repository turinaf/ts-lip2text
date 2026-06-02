"""
Preprocess the full lip-reading dataset:
- Extract 5D lip features from each video
- Segment per-digit subsequences using .lab annotations
- Split into train/test by speaker (no speaker overlap)
- Save as .npz for downstream training
"""
import cv2
import librosa
import mediapipe as mp_lib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import glob
import json
import argparse
from collections import defaultdict

# --- Configuration ---
MODEL_PATH = 'data/face_landmarker.task'
DEFAULT_DIGIT_DATA_DIR = 'data/lipdata-digit'
DEFAULT_GRID_PROCESSED_ROOT = '../liptev/data/grid'
DEFAULT_GRID_ORIGINAL_ROOT = '../data'
DEFAULT_OUTPUT_DIR = 'processed_data'

TEST_SPEAKER_RATIO = 0.1  # ~20% of speakers held out for test
RANDOM_SEED = 42

# Lip landmark indices
OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
             185, 40, 39, 37, 0, 267, 269, 270, 409]
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
             191, 80, 81, 82, 13, 312, 311, 310, 415]

IDX_TOP_INNER = 13
IDX_BOTTOM_INNER = 14
IDX_TOP_OUTER = 0
IDX_BOTTOM_OUTER = 17
IDX_LEFT_CORNER = 61
IDX_RIGHT_CORNER = 291
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

FEATURE_NAMES = ['vert_aperture', 'outer_vert_aperture', 'horiz_spread', 'inner_area', 'outer_area', 'compactness', 'lip_speed', 'rms_energy']


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess digit or GRID lipreading datasets.')
    parser.add_argument('--dataset', choices=['digit', 'grid'], default='digit', help='Dataset format to preprocess.')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output directory for train/test .npz and metadata.')
    parser.add_argument('--test-speaker-ratio', type=float, default=TEST_SPEAKER_RATIO, help='Speaker-level test split ratio.')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed for speaker split.')

    # Digit dataset options
    parser.add_argument('--digit-data-dir', default=DEFAULT_DIGIT_DATA_DIR, help='Root path of digit dataset (contains subset_*/speaker dirs).')

    # GRID dataset options
    parser.add_argument('--grid-processed-root', default=DEFAULT_GRID_PROCESSED_ROOT,
                        help='GRID processed root containing s*_processed speaker dirs with align/audio/video.')
    parser.add_argument('--grid-original-root', default=DEFAULT_GRID_ORIGINAL_ROOT,
                        help='GRID original root containing s*_processed speaker dirs with uncropped .mpg videos.')
    parser.add_argument('--grid-speakers', default='',
                        help='Comma-separated speaker IDs (e.g., s10,s11). Empty means all available speakers.')

    return parser.parse_args()


# --- Feature extraction functions ---
def extract_rms(audio_path, num_frames, fps):
    """Load audio and return per-video-frame RMS energy, shape (num_frames,)."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop_length = max(1, int(sr / fps))
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]  # (n_audio_frames,)
    # Interpolate RMS timeline onto video frame timestamps
    audio_times = np.arange(len(rms)) * hop_length / sr
    video_times = np.arange(num_frames) / fps
    return np.interp(video_times, audio_times, rms)  # (num_frames,)


def polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def polygon_perimeter(points):
    diffs = np.diff(points, axis=0, append=points[:1])
    return np.sum(np.linalg.norm(diffs, axis=1))

def extract_landmarks(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
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
    return np.array(landmarks_list), fps

def compute_features(all_lm):
    num_frames = len(all_lm)
    interocular = np.linalg.norm(
        all_lm[:, LEFT_EYE_OUTER] - all_lm[:, RIGHT_EYE_OUTER], axis=1
    )
    interocular = np.where(interocular == 0, 1.0, interocular)

    vert_ap = np.linalg.norm(
        all_lm[:, IDX_TOP_INNER] - all_lm[:, IDX_BOTTOM_INNER], axis=1
    ) / interocular
    outer_vert_ap = np.linalg.norm(
        all_lm[:, IDX_TOP_OUTER] - all_lm[:, IDX_BOTTOM_OUTER], axis=1
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

    return np.column_stack([vert_ap, outer_vert_ap, horiz_sp, inner_areas, outer_areas, compactness, lip_speed])

def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_annotation(lab_path, fps, num_frames=None):
    with open(lab_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        return [], []

    first_parts = lines[0].split()

    # GRID/HTK style alignments: "<start> <end> <token>"
    if len(first_parts) >= 3 and _is_number(first_parts[0]) and _is_number(first_parts[1]):
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
            # GRID default time unit: 1/25000 second.
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

    # Digit dataset style:
    # line 1: tokens
    # line 2: per-token time ranges "ss-es"
    if len(lines) < 2:
        return [], []

    tokens = lines[0].split()
    time_ranges = lines[1].split()
    alignments = []
    for tok, tr in zip(tokens, time_ranges):
        if '-' not in tr:
            continue
        ss, es = tr.split('-', 1)
        sf = int(float(ss) * fps)
        ef = int(float(es) * fps)
        alignments.append((sf, ef, tok))
    return tokens, alignments


def collect_digit_speaker_dirs(data_dir):
    subsets = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('subset_')])
    print(f"Found subsets: {subsets}")

    # unique_speaker_id (subset/speaker) -> list[(subset, speaker_dir)]
    speaker_dirs_map = defaultdict(list)
    for subset in subsets:
        subset_path = os.path.join(data_dir, subset)
        for spk in sorted(os.listdir(subset_path)):
            spk_path = os.path.join(subset_path, spk)
            if os.path.isdir(spk_path):
                unique_speaker_id = f"{subset}/{spk}"
                speaker_dirs_map[unique_speaker_id].append((subset, spk_path))
    return speaker_dirs_map


def collect_grid_speaker_dirs(processed_root, original_root, selected_speakers=None):
    speaker_dirs_map = defaultdict(list)
    speaker_dirs = sorted(glob.glob(os.path.join(processed_root, 's*_processed')))

    for proc_spk_dir in speaker_dirs:
        if not os.path.isdir(proc_spk_dir):
            continue
        spk_folder = os.path.basename(proc_spk_dir)  # s10_processed
        spk_id = spk_folder.replace('_processed', '')
        if selected_speakers and spk_id not in selected_speakers:
            continue

        align_dir = os.path.join(proc_spk_dir, 'align')
        audio_dir = os.path.join(proc_spk_dir, 'audio')
        if not os.path.isdir(align_dir) or not os.path.isdir(audio_dir):
            continue

        orig_spk_dir = os.path.join(original_root, spk_folder)
        if not os.path.isdir(orig_spk_dir):
            continue

        speaker_dirs_map[spk_id].append((proc_spk_dir, orig_spk_dir))

    return speaker_dirs_map


def save_split(samples, filepath):
    """Save samples as .npz — variable-length sequences stored as object arrays."""
    video_ids = []
    speakers = []
    digit_sequences = []
    full_features_list = []
    digit_segments_list = []
    fps_list = []

    for s in samples:
        video_ids.append(s['video_id'])
        speakers.append(s['speaker'])
        digit_sequences.append(s['digit_sequence'])
        full_features_list.append(s['full_features'])
        digit_segments_list.append(s['digit_segments'])
        fps_list.append(s['fps'])

    # Use object arrays for variable-length data
    full_features_arr = np.empty(len(samples), dtype=object)
    digit_segments_arr = np.empty(len(samples), dtype=object)
    for i in range(len(samples)):
        full_features_arr[i] = full_features_list[i]
        digit_segments_arr[i] = digit_segments_list[i]

    np.savez_compressed(
        filepath,
        video_ids=np.array(video_ids),
        speakers=np.array(speakers),
        digit_sequences=np.array(digit_sequences, dtype=object),
        full_features=full_features_arr,
        digit_segments=digit_segments_arr,
        fps=np.array(fps_list),
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"Saved {len(samples)} samples to {filepath}")


def main():
    # --- Main preprocessing ---
    args = parse_args()
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing face landmarker...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    if args.dataset == 'digit':
        speaker_dirs_map = collect_digit_speaker_dirs(args.digit_data_dir)
        print(f"Found {len(speaker_dirs_map)} unique speakers in digit dataset")
    else:
        selected_speakers = None
        if args.grid_speakers.strip():
            selected_speakers = {s.strip() for s in args.grid_speakers.split(',') if s.strip()}
        speaker_dirs_map = collect_grid_speaker_dirs(
            args.grid_processed_root,
            args.grid_original_root,
            selected_speakers=selected_speakers
        )
        print(f"Found {len(speaker_dirs_map)} GRID speakers")

    speakers = sorted(speaker_dirs_map.keys())
    if not speakers:
        raise RuntimeError("No speakers found. Check dataset paths and options.")

    # Split speakers into train/test
    rng = np.random.RandomState(args.seed)
    rng.shuffle(speakers)
    n_test = int(len(speakers) * args.test_speaker_ratio)
    if len(speakers) > 1 and args.test_speaker_ratio > 0 and n_test == 0:
        n_test = 1

    test_speakers = set(speakers[:n_test])
    train_speakers = set(speakers[n_test:])
    print(f"Train speakers: {len(train_speakers)}, Test speakers: {len(test_speakers)}")
    print(f"Test speakers: {sorted(test_speakers)}")

    # Storage: per-utterance segments with metadata
    all_samples = []
    failed_videos = []
    total_videos = 0

    for speaker in sorted(train_speakers | test_speakers):
        source_entries = speaker_dirs_map[speaker]

        for source_a, source_b in source_entries:
            if args.dataset == 'digit':
                subset, speaker_dir = source_a, source_b
                lab_dir = os.path.join(speaker_dir, 'lab')
                video_dir = os.path.join(speaker_dir, 'video')
                audio_dir = os.path.join(speaker_dir, 'audio')
                lab_ext = '.lab'
                video_ext = '.mp4'
                video_prefix = f"{subset}/{speaker}"
            else:
                processed_speaker_dir, original_speaker_dir = source_a, source_b
                lab_dir = os.path.join(processed_speaker_dir, 'align')
                audio_dir = os.path.join(processed_speaker_dir, 'audio')
                video_dir = original_speaker_dir
                lab_ext = '.align'
                video_ext = '.mpg'
                video_prefix = f"grid/{speaker}"

            if not os.path.isdir(lab_dir) or not os.path.isdir(video_dir):
                continue

            lab_files = sorted(glob.glob(os.path.join(lab_dir, '*' + lab_ext)))

            for lab_path in lab_files:
                base_name = os.path.splitext(os.path.basename(lab_path))[0]
                video_path = os.path.join(video_dir, base_name + video_ext)
                audio_path = os.path.join(audio_dir, base_name + '.wav')
                video_id = f"{video_prefix}/{base_name}"

                if not os.path.exists(video_path):
                    failed_videos.append((video_id, "video not found"))
                    continue

                total_videos += 1
                if total_videos % 50 == 0:
                    print(f"  Processing video {total_videos}...")

                try:
                    all_lm, fps = extract_landmarks(video_path, detector)
                    if all_lm is None or len(all_lm) < 5:
                        failed_videos.append((video_id, "landmark extraction failed"))
                        continue

                    visual_features = compute_features(all_lm)
                    if os.path.exists(audio_path):
                        rms = extract_rms(audio_path, len(all_lm), fps)
                    else:
                        rms = np.zeros(len(all_lm), dtype=np.float32)
                    features = np.column_stack([visual_features, rms])
                    token_seq, alignments = parse_annotation(lab_path, fps, num_frames=len(all_lm))
                    if not alignments:
                        failed_videos.append((video_id, "empty alignment"))
                        continue

                    # Extract per-token segments
                    token_segments = []
                    valid = True
                    for sf, ef, _ in alignments:
                        sf_c = min(sf, features.shape[0])
                        ef_c = min(ef, features.shape[0])
                        if ef_c - sf_c < 2:
                            valid = False
                            break
                        token_segments.append(features[sf_c:ef_c])

                    if not valid:
                        failed_videos.append((video_id, "segment too short"))
                        continue

                    split = 'test' if speaker in test_speakers else 'train'
                    all_samples.append({
                        'video_id': video_id,
                        'speaker': speaker,
                        'split': split,
                        'digit_sequence': token_seq,
                        'full_features': features,
                        'digit_segments': token_segments,
                        'alignments': alignments,
                        'fps': fps,
                    })

                except Exception as e:
                    failed_videos.append((video_id, str(e)))
                    continue

    print(f"\nProcessed {total_videos} videos total")
    print(f"Successful: {len(all_samples)}, Failed: {len(failed_videos)}")
    if failed_videos:
        print(f"First 10 failures:")
        for vid, reason in failed_videos[:10]:
            print(f"  {vid}: {reason}")

    # --- Save processed data ---
    train_samples = [s for s in all_samples if s['split'] == 'train']
    test_samples = [s for s in all_samples if s['split'] == 'test']

    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Count token distribution
    train_digit_count = defaultdict(int)
    test_digit_count = defaultdict(int)
    for s in train_samples:
        for d in s['digit_sequence']:
            train_digit_count[d] += 1
    for s in test_samples:
        for d in s['digit_sequence']:
            test_digit_count[d] += 1

    print(f"\nTrain digit distribution: {dict(sorted(train_digit_count.items()))}")
    print(f"Test digit distribution:  {dict(sorted(test_digit_count.items()))}")

    save_split(train_samples, os.path.join(output_dir, 'train.npz'))
    save_split(test_samples, os.path.join(output_dir, 'test.npz'))

    metadata = {
        'dataset': args.dataset,
        'n_features': len(FEATURE_NAMES),
        'feature_names': FEATURE_NAMES,
        'n_train': len(train_samples),
        'n_test': len(test_samples),
        'train_speakers': sorted(train_speakers),
        'test_speakers': sorted(test_speakers),
        'train_digit_dist': dict(sorted(train_digit_count.items())),
        'test_digit_dist': dict(sorted(test_digit_count.items())),
        'n_failed': len(failed_videos),
        'digits_per_video': 8 if args.dataset == 'digit' else None,
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {os.path.join(output_dir, 'metadata.json')}")

    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()
