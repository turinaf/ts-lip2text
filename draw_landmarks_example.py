"""
Draw and label key facial landmarks used in feature extraction on a single frame.
Saves the annotated frame as 'annotated_frame.png'.

"""
import cv2
import mediapipe as mp_lib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# --- Config ---
MODEL_PATH = 'data/face_landmarker.task'
VIDEO_PATH = 'data/lipdata-digit/subset_05/1038/video/1038_male_iPhone_iOS10.1_20161117143619_30755876.mp4'
OUTPUT_IMAGE = 'annotated_frame2.png'

# Landmark indices (from preprocess.py)
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

# --- Load model ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- Read a single frame ---
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError('Failed to read frame from video')

h, w = frame.shape[:2]
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
result = detector.detect(mp_image)

if not result.face_landmarks:
    raise RuntimeError('No face landmarks detected')

face_lm = result.face_landmarks[0]
lm = np.array([[l.x * w, l.y * h] for l in face_lm])


# --- Professional landmark visualization ---
annot = frame.copy()

# Draw and connect outer lip (blue, thin, anti-aliased)
outer_pts = lm[OUTER_LIP].astype(np.int32)
for i in range(len(outer_pts)):
    pt1 = tuple(outer_pts[i])
    pt2 = tuple(outer_pts[(i + 1) % len(outer_pts)])
    cv2.line(annot, pt1, pt2, (70, 120, 255), 1, cv2.LINE_AA)
for pt in outer_pts:
    cv2.circle(annot, tuple(pt), 3, (70, 120, 255), -1, cv2.LINE_AA)

# Draw and connect inner lip (cyan, thin, anti-aliased)
inner_pts = lm[INNER_LIP].astype(np.int32)
for i in range(len(inner_pts)):
    pt1 = tuple(inner_pts[i])
    pt2 = tuple(inner_pts[(i + 1) % len(inner_pts)])
    cv2.line(annot, pt1, pt2, (0, 220, 220), 1, cv2.LINE_AA)
for pt in inner_pts:
    cv2.circle(annot, tuple(pt), 2, (0, 220, 220), -1, cv2.LINE_AA)

# Draw key anatomical points (red, larger) and annotate with p{idx}
keypoints = [
    IDX_TOP_OUTER, IDX_BOTTOM_OUTER, IDX_LEFT_CORNER, IDX_RIGHT_CORNER,
    IDX_TOP_INNER, IDX_BOTTOM_INNER,
    LEFT_EYE_OUTER, RIGHT_EYE_OUTER,
]
for idx in keypoints:
    pt = tuple(np.round(lm[idx]).astype(int))
    cv2.circle(annot, pt, 3, (0, 0, 255), -1, cv2.LINE_AA)  # smaller dot
    # Smaller font size for label
    cv2.putText(annot, f'p{idx}', (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(annot, f'p{idx}', (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

# Draw inter-ocular line (magenta, thin, anti-aliased)
pt1 = tuple(np.round(lm[LEFT_EYE_OUTER]).astype(int))
pt2 = tuple(np.round(lm[RIGHT_EYE_OUTER]).astype(int))
cv2.line(annot, pt1, pt2, (255, 0, 255), 1, cv2.LINE_AA)

cv2.imwrite(OUTPUT_IMAGE, annot)
print(f'Annotated frame saved as {OUTPUT_IMAGE}')
