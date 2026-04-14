"""
Compare feature visualizations across multiple video samples.
Processes 5 videos from different speakers and generates per-video feature overview plots.
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# --- Configuration ---
MODEL_PATH = 'data/face_landmarker.task'
DATA_DIR = 'data/lipdata0405-filter'
OUTPUT_DIR = 'output_multi_sample'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pick one video from each of 5 different speakers
SPEAKERS = ['1003', '1004', '1006', '1011', '1013']

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

FEATURE_NAMES = [
    'Vertical Aperture',
    'Horizontal Spread',
    'Inner Lip Area',
    'Compactness',
    'Lip Speed',
]


# --- Helper functions ---
def polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def polygon_perimeter(points):
    diffs = np.diff(points, axis=0, append=points[:1])
    return np.sum(np.linalg.norm(diffs, axis=1))


def extract_landmarks(video_path, detector):
    """Extract all face landmarks from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path}")
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
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
    """Compute the 5 features from face landmarks array (T, 478, 2)."""
    num_frames = len(all_lm)

    interocular = np.linalg.norm(
        all_lm[:, LEFT_EYE_OUTER] - all_lm[:, RIGHT_EYE_OUTER], axis=1
    )
    interocular = np.where(interocular == 0, 1.0, interocular)

    top_inner = all_lm[:, IDX_TOP_INNER]
    bot_inner = all_lm[:, IDX_BOTTOM_INNER]
    left_c = all_lm[:, IDX_LEFT_CORNER]
    right_c = all_lm[:, IDX_RIGHT_CORNER]

    vert_ap = np.linalg.norm(top_inner - bot_inner, axis=1) / interocular
    horiz_sp = np.linalg.norm(left_c - right_c, axis=1) / interocular

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
    """Parse .lab annotation file. Returns list of (start_frame, end_frame, digit)."""
    with open(lab_path) as f:
        lines = f.read().strip().split('\n')
    digits = lines[0].strip().split()
    time_ranges = lines[1].strip().split()

    alignments = []
    for digit, tr in zip(digits, time_ranges):
        start_sec, end_sec = tr.split('-')
        sf = int(float(start_sec) * fps)
        ef = int(float(end_sec) * fps)
        alignments.append((sf, ef, digit))
    return alignments


def plot_features_timeline(features, feature_names, alignments, fps, num_frames, title, save_path):
    """Plot all features with digit segments highlighted."""
    n_feat = features.shape[1]
    fig, axes = plt.subplots(n_feat, 1, figsize=(14, 2.5 * n_feat), sharex=True)
    time_axis = np.arange(num_frames) / fps

    unique_digits = list(dict.fromkeys(d for _, _, d in alignments))
    cmap = plt.get_cmap('tab10')
    colors = {d: cmap(i % 10) for i, d in enumerate(unique_digits)}

    for ax, fname, col in zip(axes, feature_names, range(n_feat)):
        signal = features[:, col]
        ax.plot(time_axis, signal, color='lightgray', linewidth=0.8, zorder=1)

        plotted = set()
        for sf, ef, digit in alignments:
            sf_c, ef_c = min(sf, len(signal)), min(ef, len(signal))
            if sf_c < ef_c:
                t = time_axis[sf_c:ef_c]
                s = signal[sf_c:ef_c]
                label = digit if digit not in plotted else ""
                plotted.add(digit)
                ax.plot(t, s, color=colors[digit], linewidth=2, label=label, zorder=2)
                mid_t = (t[0] + t[-1]) / 2
                y_range = np.max(signal) - np.min(signal)
                ax.text(mid_t, np.max(s) + 0.02 * (y_range + 1e-8),
                        digit, color=colors[digit], fontsize=9, fontweight='bold', ha='center')

        ax.set_ylabel(fname, fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# --- Main: process multiple samples ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

sample_results = []  # store (speaker, digits_str, features, alignments, fps) for summary

for speaker in SPEAKERS:
    speaker_dir = os.path.join(DATA_DIR, speaker)
    lab_files = sorted(glob.glob(os.path.join(speaker_dir, '*.lab')))
    if not lab_files:
        print(f"No .lab files for speaker {speaker}, skipping")
        continue

    # Take the first video for this speaker
    lab_path = lab_files[0]
    base = lab_path.replace('.lab', '')
    video_path = base + '.mp4'

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}, skipping")
        continue

    print(f"\n{'='*60}")
    print(f"Processing: Speaker {speaker}")
    print(f"  Video: {video_path}")
    print(f"  Annotation: {lab_path}")

    # Extract & compute
    all_lm, fps = extract_landmarks(video_path, detector)
    if all_lm is None:
        continue

    features = compute_features(all_lm)
    alignments = parse_annotation(lab_path, fps)
    num_frames = len(all_lm)

    # Read digit sequence for title
    with open(lab_path) as f:
        digits_str = f.readline().strip()

    print(f"  Frames: {num_frames}, FPS: {fps}")
    print(f"  Digits: {digits_str}")
    for sf, ef, d in alignments:
        print(f"    '{d}': frames {sf}-{ef}")

    # Plot
    title = f'Speaker {speaker} — Digits: {digits_str}'
    save_path = os.path.join(OUTPUT_DIR, f'features_{speaker}.png')
    plot_features_timeline(features, FEATURE_NAMES, alignments, fps, num_frames, title, save_path)
    print(f"  Saved: {save_path}")

    sample_results.append((speaker, digits_str, features, alignments, fps, num_frames))


# --- Summary: side-by-side comparison of same digit across speakers ---
print(f"\n{'='*60}")
print("Cross-speaker digit comparison")

# Collect all digit occurrences: digit -> [(speaker, segment_features)]
from collections import defaultdict
digit_segments = defaultdict(list)

for speaker, digits_str, features, alignments, fps, num_frames in sample_results:
    for sf, ef, digit in alignments:
        sf_c, ef_c = min(sf, features.shape[0]), min(ef, features.shape[0])
        if sf_c < ef_c:
            digit_segments[digit].append((speaker, features[sf_c:ef_c]))

# Plot digits that appear in at least 2 speakers
common_digits = sorted([d for d, segs in digit_segments.items() if len(segs) >= 2])
if common_digits:
    n_digits = len(common_digits)
    n_feats = len(FEATURE_NAMES)

    fig, axes = plt.subplots(n_digits, n_feats,
                             figsize=(3.5 * n_feats, 2.5 * n_digits),
                             squeeze=False)

    cmap = plt.get_cmap('Set1')
    speaker_colors = {s: cmap(i % 9) for i, s in enumerate(SPEAKERS)}

    for row, digit in enumerate(common_digits):
        segs = digit_segments[digit]
        for col in range(n_feats):
            ax = axes[row, col]
            for speaker, seg_feat in segs:
                t_norm = np.linspace(0, 1, len(seg_feat))
                ax.plot(t_norm, seg_feat[:, col], linewidth=1.5, alpha=0.7,
                        color=speaker_colors[speaker], label=speaker)
            if row == 0:
                ax.set_title(FEATURE_NAMES[col], fontsize=8)
            if col == 0:
                ax.set_ylabel(f'Digit {digit}', fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(labelsize=7)

    # Deduplicated legend
    handles, labels = axes[0, -1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9, title='Speaker')

    plt.suptitle('Same Digit Across Different Speakers (time normalized)', fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'cross_speaker_digit_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
else:
    print("No common digits found across speakers.")

print(f"\nAll outputs in: {OUTPUT_DIR}/")
