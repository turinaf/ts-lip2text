"""Find the maximum frame count across all videos in the dataset."""

import os
import json
import cv2
import glob

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    dataset = config["dataset"]
    if dataset == "grid":
        data_dir = config["grid_processed_data_dir"]
    else:
        data_dir = config["digit_processed_data_dir"]

    video_files = glob.glob(os.path.join(data_dir, "**", "*.mp4"), recursive=True)

    if not video_files:
        print(f"No .mp4 files found in {data_dir}")
        return

    max_frames = 0
    max_file = ""
    total = len(video_files)

    for i, vf in enumerate(video_files, 1):
        frames = count_frames(vf)
        if frames > max_frames:
            max_frames = frames
            max_file = vf
        if i % 500 == 0 or i == total:
            print(f"Processed {i}/{total} videos...")

    print(f"\nMax frame count: {max_frames}")
    print(f"File: {max_file}")
    print(f"Total videos scanned: {total}")

if __name__ == "__main__":
    main()
