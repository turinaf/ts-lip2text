"""
Preprocess the full lip-reading dataset:
- Extract 5D lip features from each video
- Segment per-digit subsequences using .lab annotations
- Split into train/test by speaker (no speaker overlap)
- Save as .npz for downstream training
"""
import cv2
import mediapipe as mp_lib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import glob
import json
from collections import defaultdict

# --- Configuration ---
MODEL_PATH = 'data/face_landmarker.task'
DATA_DIR = 'data/lipdata0405-filter'
OUTPUT_DIR = 'processed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SPEAKER_RATIO = 0.2  # ~20% of speakers held out for test
RANDOM_SEED = 42

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

FEATURE_NAMES = ['vert_aperture', 'horiz_spread', 'inner_area', 'compactness', 'lip_speed']


# --- Feature extraction functions ---
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

def parse_annotation(lab_path, fps):
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


# --- Main preprocessing ---
print("Initializing face landmarker...")
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Discover all speakers
speakers = sorted([d for d in os.listdir(DATA_DIR)
                    if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"Found {len(speakers)} speakers")

# Split speakers into train/test
rng = np.random.RandomState(RANDOM_SEED)
rng.shuffle(speakers)
n_test = max(1, int(len(speakers) * TEST_SPEAKER_RATIO))
test_speakers = set(speakers[:n_test])
train_speakers = set(speakers[n_test:])
print(f"Train speakers: {len(train_speakers)}, Test speakers: {len(test_speakers)}")
print(f"Test speakers: {sorted(test_speakers)}")

# Storage: per-digit segments with metadata
# Each sample: {features: (T, 5), digit_sequence: [str], speaker: str, video_id: str}
# We store both:
#   1) Full-video features + digit sequence (for sequence-level verification)
#   2) Per-digit segments (for digit-level analysis)

all_samples = []  # full video level
failed_videos = []
total_videos = 0

for speaker in sorted(train_speakers | test_speakers):
    speaker_dir = os.path.join(DATA_DIR, speaker)
    lab_files = sorted(glob.glob(os.path.join(speaker_dir, '*.lab')))

    for lab_path in lab_files:
        base = lab_path.replace('.lab', '')
        video_path = base + '.mp4'
        video_id = os.path.basename(base)

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

            features = compute_features(all_lm)
            digit_seq, alignments = parse_annotation(lab_path, fps)

            # Extract per-digit segments
            digit_segments = []
            valid = True
            for sf, ef, digit in alignments:
                sf_c = min(sf, features.shape[0])
                ef_c = min(ef, features.shape[0])
                if ef_c - sf_c < 2:
                    valid = False
                    break
                digit_segments.append(features[sf_c:ef_c])

            if not valid:
                failed_videos.append((video_id, "segment too short"))
                continue

            split = 'test' if speaker in test_speakers else 'train'
            all_samples.append({
                'video_id': video_id,
                'speaker': speaker,
                'split': split,
                'digit_sequence': digit_seq,
                'full_features': features,          # (T_video, 5)
                'digit_segments': digit_segments,   # list of (T_digit, 5)
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

# Count digit distribution
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


save_split(train_samples, os.path.join(OUTPUT_DIR, 'train.npz'))
save_split(test_samples, os.path.join(OUTPUT_DIR, 'test.npz'))

# Save metadata
metadata = {
    'n_features': len(FEATURE_NAMES),
    'feature_names': FEATURE_NAMES,
    'n_train': len(train_samples),
    'n_test': len(test_samples),
    'train_speakers': sorted(train_speakers),
    'test_speakers': sorted(test_speakers),
    'train_digit_dist': dict(sorted(train_digit_count.items())),
    'test_digit_dist': dict(sorted(test_digit_count.items())),
    'n_failed': len(failed_videos),
    'digits_per_video': 8,
}
with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata to {os.path.join(OUTPUT_DIR, 'metadata.json')}")

print("\nPreprocessing complete!")
