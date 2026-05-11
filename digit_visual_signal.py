import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration ---
video_path = 'data/lipdata-digit/subset_05/1003/video/1003_20161115142134_35396787.mp4'
model_path = 'data/face_landmarker.task'
annotation_path = 'data/lipdata-digit/subset_05/1003/lab/1003_20161115142134_35396787.lab'

# Lip landmark indices (MediaPipe face mesh)
OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
             185, 40, 39, 37, 0, 267, 269, 270, 409]
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
             191, 80, 81, 82, 13, 312, 311, 310, 415]

# Key anatomical landmark indices (MediaPipe face mesh)
IDX_TOP_OUTER = 0       # top center of outer lip
IDX_BOTTOM_OUTER = 17   # bottom center of outer lip
IDX_LEFT_CORNER = 61    # left corner
IDX_RIGHT_CORNER = 291  # right corner
IDX_TOP_INNER = 13      # top center of inner lip
IDX_BOTTOM_INNER = 14   # bottom center of inner lip

# Normalization: inter-ocular distance landmarks
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# --- 2. Extract Face Landmarks ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {video_path}")
print(f"FPS: {fps}, Resolution: {vid_width}x{vid_height}, Total frames: {total_frames}")

all_face_landmarks_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        face_lm = detection_result.face_landmarks[0]
        all_lm = np.array([[lm.x * vid_width, lm.y * vid_height] for lm in face_lm])
        all_face_landmarks_list.append(all_lm)
    else:
        if all_face_landmarks_list:
            all_face_landmarks_list.append(all_face_landmarks_list[-1])
        else:
            all_face_landmarks_list.append(np.zeros((478, 2)))

cap.release()
all_face_landmarks = np.array(all_face_landmarks_list)  # (T, 478, 2)
num_frames = len(all_face_landmarks)
print(f"Extracted landmarks for {num_frames} frames")


# --- 3. Normalization: inter-ocular distance (head-size invariant) ---
interocular = np.linalg.norm(
    all_face_landmarks[:, LEFT_EYE_OUTER] - all_face_landmarks[:, RIGHT_EYE_OUTER],
    axis=1
)
interocular = np.where(interocular == 0, 1.0, interocular)


# --- 4. Feature Extraction ---
def polygon_area(points):
    """Shoelace formula for polygon area."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def polygon_perimeter(points):
    """Perimeter of a polygon defined by ordered points."""
    diffs = np.diff(points, axis=0, append=points[:1])
    return np.sum(np.linalg.norm(diffs, axis=1))

def compute_angle(p1, vertex, p2):
    """Angle at vertex formed by p1-vertex-p2, in degrees."""
    v1, v2 = p1 - vertex, p2 - vertex
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

# Pre-extract key landmark positions for all frames
top_outer = all_face_landmarks[:, IDX_TOP_OUTER]
bot_outer = all_face_landmarks[:, IDX_BOTTOM_OUTER]
left_corner = all_face_landmarks[:, IDX_LEFT_CORNER]
right_corner = all_face_landmarks[:, IDX_RIGHT_CORNER]
top_inner = all_face_landmarks[:, IDX_TOP_INNER]
bot_inner = all_face_landmarks[:, IDX_BOTTOM_INNER]

# Geometric features (normalized by inter-ocular distance)
vert_aperture = np.linalg.norm(top_inner - bot_inner, axis=1) / interocular
horiz_spread = np.linalg.norm(left_corner - right_corner, axis=1) / interocular

# Area, perimeter, compactness (per-frame)
inner_areas = np.zeros(num_frames)
outer_compactness = np.zeros(num_frames)

for i in range(num_frames):
    lm = all_face_landmarks[i]
    outer_pts, inner_pts = lm[OUTER_LIP], lm[INNER_LIP]

    oa = polygon_area(outer_pts)
    ia = polygon_area(inner_pts)
    op = polygon_perimeter(outer_pts)

    norm2 = interocular[i] ** 2
    inner_areas[i] = ia / norm2
    outer_compactness[i] = (4 * np.pi * oa) / (op ** 2 + 1e-8)

# Dynamic feature: lip speed (magnitude of velocity vector)
vert_velocity = np.gradient(vert_aperture)
horiz_velocity = np.gradient(horiz_spread)
lip_speed = np.sqrt(vert_velocity ** 2 + horiz_velocity ** 2)

feature_names = [
    'Vertical Aperture',    # 0 - inner lip opening (most speech-relevant)
    'Horizontal Spread',    # 1 - how wide the lips stretch
    'Inner Lip Area',       # 2 - oral opening area
    'Compactness',          # 3 - circular (rounded) vs elongated opening
    'Lip Speed',            # 4 - overall speed of lip movement
]

features = np.column_stack([
    vert_aperture, horiz_spread,
    inner_areas, outer_compactness,
    lip_speed,
])
print(f"Feature matrix: {features.shape} ({num_frames} frames × {len(feature_names)} features)")


# --- 5. Parse annotation (.lab) file ---
# Format: line 1 = digits, line 2 = start-end times in seconds
with open(annotation_path) as f:
    lines = f.read().strip().split('\n')

digits = lines[0].strip().split()
time_ranges = lines[1].strip().split()

alignments = []
for digit, time_range in zip(digits, time_ranges):
    start_sec, end_sec = time_range.split('-')
    start_frame = int(float(start_sec) * fps)
    end_frame = int(float(end_sec) * fps)
    alignments.append((start_frame, end_frame, digit))

print(f"\nAnnotations ({len(alignments)} digits):")
for sf, ef, d in alignments:
    print(f"  Digit '{d}': frames {sf}-{ef} ({(ef-sf)/fps:.2f}s)")

frame_labels = ['sil'] * num_frames
for sf, ef, digit in alignments:
    for i in range(sf, min(ef, num_frames)):
        frame_labels[i] = digit


# --- 6. Plot 1: All features over time with digit segments highlighted ---
def plot_features_timeline(features, feature_names, alignments, fps, num_frames):
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
    axes[0].set_title('Lip Movement Features Over Time with Digit Annotations')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig('features_overview.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_features_timeline(features, feature_names, alignments, fps, num_frames)


# --- 7. Plot 2: Per-digit comparison (overlay all occurrences of same digit) ---
def plot_digit_comparison(features, feature_names, alignments):
    unique_digits = sorted(set(d for _, _, d in alignments))
    key_feats = [0, 1, 2, 3, 4]  # all features

    fig, axes = plt.subplots(len(unique_digits), len(key_feats),
                             figsize=(3.5 * len(key_feats), 2.5 * len(unique_digits)),
                             squeeze=False)

    for row, digit in enumerate(unique_digits):
        segments = [(sf, ef) for sf, ef, d in alignments if d == digit]
        for col, fi in enumerate(key_feats):
            ax = axes[row, col]
            for si, (sf, ef) in enumerate(segments):
                sf_c, ef_c = min(sf, features.shape[0]), min(ef, features.shape[0])
                if sf_c < ef_c:
                    seg = features[sf_c:ef_c, fi]
                    t_norm = np.linspace(0, 1, len(seg))
                    ax.plot(t_norm, seg, linewidth=1.5, alpha=0.8, label=f'occ {si+1}')
            if row == 0:
                ax.set_title(feature_names[fi], fontsize=8)
            if col == 0:
                ax.set_ylabel(f'Digit {digit}', fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(labelsize=7)

    plt.suptitle('Per-Digit Feature Comparison (time normalized to [0,1])', fontsize=12)
    plt.tight_layout()
    plt.savefig('digit_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_digit_comparison(features, feature_names, alignments)


# --- 8. Plot 3: Cross-digit comparison ---
def plot_cross_digit(features, feature_names, alignments):
    unique_digits = sorted(set(d for _, _, d in alignments))
    key_feats = [0, 1, 2, 3]
    cmap = plt.get_cmap('tab10')
    colors = {d: cmap(i % 10) for i, d in enumerate(unique_digits)}

    fig, axes = plt.subplots(len(key_feats), 1, figsize=(10, 3 * len(key_feats)))
    for ax, fi in zip(axes, key_feats):
        for sf, ef, digit in alignments:
            sf_c, ef_c = min(sf, features.shape[0]), min(ef, features.shape[0])
            if sf_c < ef_c:
                seg = features[sf_c:ef_c, fi]
                t_norm = np.linspace(0, 1, len(seg))
                ax.plot(t_norm, seg, color=colors[digit], linewidth=1.5, alpha=0.7, label=digit)
        ax.set_ylabel(feature_names[fi], fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')

    axes[-1].set_xlabel('Normalized Time')
    axes[0].set_title('Cross-Digit Feature Comparison')
    plt.tight_layout()
    plt.savefig('cross_digit_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_cross_digit(features, feature_names, alignments)


# --- 9. Generate video with lip landmark overlay ---
def generate_landmark_video(video_path, all_face_landmarks, alignments, fps, output_path='lip_landmarks_overlay.mp4'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Build frame→digit lookup
    frame_digit = {}
    for sf, ef, digit in alignments:
        for i in range(sf, ef):
            frame_digit[i] = digit

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(all_face_landmarks):
            lm = all_face_landmarks[frame_idx]

            # Draw outer lip contour (green)
            outer_pts = lm[OUTER_LIP].astype(np.int32)
            for i in range(len(outer_pts)):
                pt1 = tuple(outer_pts[i])
                pt2 = tuple(outer_pts[(i + 1) % len(outer_pts)])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            for pt in outer_pts:
                cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

            # Draw inner lip contour (cyan)
            inner_pts = lm[INNER_LIP].astype(np.int32)
            for i in range(len(inner_pts)):
                pt1 = tuple(inner_pts[i])
                pt2 = tuple(inner_pts[(i + 1) % len(inner_pts)])
                cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
            for pt in inner_pts:
                cv2.circle(frame, tuple(pt), 3, (255, 255, 0), -1)

            # Draw key anatomical points (red, larger)
            for idx in [IDX_TOP_OUTER, IDX_BOTTOM_OUTER, IDX_LEFT_CORNER, IDX_RIGHT_CORNER,
                        IDX_TOP_INNER, IDX_BOTTOM_INNER]:
                pt = tuple(lm[idx].astype(np.int32))
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # Draw vertical aperture line (outer: yellow dashed)
            cv2.line(frame, tuple(lm[IDX_TOP_OUTER].astype(np.int32)),
                     tuple(lm[IDX_BOTTOM_OUTER].astype(np.int32)), (0, 255, 255), 1)
            # Draw horizontal spread line (magenta)
            cv2.line(frame, tuple(lm[IDX_LEFT_CORNER].astype(np.int32)),
                     tuple(lm[IDX_RIGHT_CORNER].astype(np.int32)), (255, 0, 255), 1)

        # Overlay digit label and frame number
        digit_label = frame_digit.get(frame_idx, 'sil')
        cv2.putText(frame, f'Frame {frame_idx}  Digit: {digit_label}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Legend
        cv2.putText(frame, 'Green: Outer lip', (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, 'Cyan: Inner lip', (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, 'Red: Key points', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Landmark overlay video saved to: {output_path}")

generate_landmark_video(video_path, all_face_landmarks, alignments, fps)

print("\nSaved: features_overview.png, digit_comparison.png, cross_digit_comparison.png, lip_landmarks_overlay.mp4")