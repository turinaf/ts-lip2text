import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Lip landmark indices (your provided list)
LIP_LANDMARKS = [
    61,146,91,181,84,17,314,405,321,375,291,
    185,40,39,37,0,267,269,270,409,
    78,95,88,178,87,14,317,402,318,324,308
]

# Initialize FaceLandmarker
model_path = "data/face_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Open video
cap = cv2.VideoCapture('data/bbaf2n.mpg')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('data/lip_landmarks_annotated.mp4', fourcc, fps, (width, height))

lip_landmarks_time_series = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)
    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        lips = np.array([[lm.x * width, lm.y * height] for idx, lm in enumerate(face_landmarks) if idx in LIP_LANDMARKS])
        lip_landmarks_time_series.append(lips)
        # Draw landmarks on frame
        for (x, y) in lips:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    out.write(frame)
cap.release()
out.release()

lip_landmarks_time_series = np.array(lip_landmarks_time_series)  # (frames, num_lip_landmarks, 2)

# Save the time series for later use
np.save('data/lip_landmarks_time_series.npy', lip_landmarks_time_series)

# Example visualization: plot y trajectory of landmark 61 (first in your list)
plt.plot(lip_landmarks_time_series[:, 0, 1])
plt.title('Landmark 61 Y trajectory')
plt.xlabel('Frame')
plt.ylabel('Normalized Y')
plt.show()

# Optional: visualize the lip shape for a specific frame
frame_idx = 0  # change as needed
plt.scatter(lip_landmarks_time_series[frame_idx, :, 0], lip_landmarks_time_series[frame_idx, :, 1])
plt.title(f'Lip landmarks at frame {frame_idx}')
plt.gca().invert_yaxis()
plt.show()

# --- Animation of Lip Landmarks ---
fig, ax = plt.subplots()
sc = ax.scatter([], [])
ax.set_xlim(0, width)
ax.set_ylim(height, 0)  # Invert y-axis for image coordinates

def update(frame_idx):
    sc.set_offsets(lip_landmarks_time_series[frame_idx])
    ax.set_title(f'Frame {frame_idx}')
    return sc,

ani = FuncAnimation(fig, update, frames=len(lip_landmarks_time_series), interval=1000/fps, blit=True)
plt.show()

# Save animation as GIF
ani.save('data/lip_landmarks_animation.gif', writer='pillow', fps=fps)



# Parse align file
alignments = []
with open('data/bbaf2n.align') as f:
    for line in f:
        start, end, word = line.strip().split()
        alignments.append((int(start), int(end), word))

# Build frame-to-word mapping
frame_labels = []
for frame_idx in range(len(lip_landmarks_time_series)):
    frame_sample = int(frame_idx * 25000 / fps)
    label = 'sil'  # default to silence
    for start, end, word in alignments:
        if start <= frame_sample < end:
            label = word
            break
    frame_labels.append(label)

# Plot the y trajectory of a landmark, colored by word
colors = {'sil': 'gray', 'bin': 'red', 'blue': 'blue', 'at': 'green', 'f': 'orange', 'two': 'purple', 'now': 'brown'}
y_traj = lip_landmarks_time_series[:, 0, 1]
plt.figure(figsize=(12, 4))
for i, (y, label) in enumerate(zip(y_traj, frame_labels)):
    plt.plot(i, y, '.', color=colors.get(label, 'black'))
plt.xlabel('Frame')
plt.ylabel('Landmark 61 Y')
plt.title('Landmark 61 Y trajectory colored by word')
plt.show()