import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt

# Lip landmark indices 
LIP_LANDMARKS = [
	61,146,91,181,84,17,314,405,321,375,291,
	185,40,39,37,0,267,269,270,409,
	78,95,88,178,87,14,317,402,318,324,308
]

# Load precomputed lip_landmarks_time_series (frames, num_lip_landmarks, 2)
# For demonstration, you should load this from a .npy file or recompute as in your previous script
lip_landmarks_time_series = np.load('data/lip_landmarks_time_series.npy')  # Save this in your extraction script

# Get video properties
cap = cv2.VideoCapture('data/1_2003_female_20250701092055_49427391_iPhone13,2-iOS17.5_train_1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

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

# Indices for height/width calculation (using LIP_LANDMARKS positions)

# Use correct indices: 0 (top), 17 (bottom), 61 (left), 291 (right)
top_idx = LIP_LANDMARKS.index(0)
bottom_idx = LIP_LANDMARKS.index(17)
left_idx = LIP_LANDMARKS.index(61)
right_idx = LIP_LANDMARKS.index(291)

# Compute ratio for each frame
ratios = []
for lips in lip_landmarks_time_series:
	top = lips[top_idx]
	bottom = lips[bottom_idx]
	left = lips[left_idx]
	right = lips[right_idx]
	height = np.linalg.norm(top - bottom)
	width = np.linalg.norm(left - right)
	ratio = height / width if width != 0 else 0
	ratios.append(ratio)
ratios = np.array(ratios)

# Plot ratio over time for each word

# Use subplots for each word and collect the plotted ratio series
unique_words = [w for w in sorted(set(frame_labels)) if w != 'sil']
num_words = len(unique_words)
fig, axes = plt.subplots(num_words, 1, figsize=(12, 3*num_words), sharex=True)
if num_words == 1:
	axes = [axes]

word_ratios = {}
for ax, word in zip(axes, unique_words):
	indices = [i for i, label in enumerate(frame_labels) if label == word]
	series = ratios[indices]
	word_ratios[word] = series
	ax.plot(range(len(series)), series, marker='o', linestyle='-')
	ax.set_ylabel('Ratio')
	ax.set_title(f'Word: {word}')
axes[-1].set_xlabel('Frame (relative to word start)')
plt.tight_layout()
plt.show()
