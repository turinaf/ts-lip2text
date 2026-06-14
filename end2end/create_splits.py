import os
import glob
import json
import random

def create_splits(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    dataset = config.get("dataset")
    
    if dataset == "grid":
        out_dir = config.get("grid_processed_data_dir", "./data/grid")
        # Grid structure: data/grid/speaker/video/*.mp4
        search_pattern = os.path.join(out_dir, '*', 'video', '*.mp4')
    else:
        out_dir = config.get("digit_processed_data_dir", "./data/digit")
        # Digit structure: data/digit/subset_X/speaker/video/*.mp4
        search_pattern = os.path.join(out_dir, '*', '*', 'video', '*.mp4')
        
    print(f"\nCreating Train/Val/Test splits for {dataset.upper()} dataset...")
    
    # Use recursive matching just in case if glob issues persist, but explicit pattern is safer
    all_videos = glob.glob(search_pattern)
    
    if not all_videos:
        print(f"Failed to find videos! Checked pattern: {search_pattern}")
        return
    
    # Convert backslashes to forward slashes for WSL / cross-platform consistency
    all_videos = [v.replace('\\', '/') for v in all_videos]
    
    random.seed(42) 
    random.shuffle(all_videos)
    
    total_size = len(all_videos)
    test_size = max(1, int(total_size * 0.1))
    val_size = max(1, int(total_size * 0.1))
    train_size = total_size - val_size - test_size
    
    train_videos = all_videos[:train_size]
    val_videos = all_videos[train_size:train_size + val_size]
    test_videos = all_videos[train_size + val_size:]
    
    with open(os.path.join(out_dir, "train_split.json"), "w") as f:
        json.dump(train_videos, f)
    with open(os.path.join(out_dir, "val_split.json"), "w") as f:
        json.dump(val_videos, f)
    with open(os.path.join(out_dir, "test_split.json"), "w") as f:
        json.dump(test_videos, f)
        
    print(f"Data split saved in {out_dir}:")
    print(f" - Train: {len(train_videos)}")
    print(f" - Val:   {len(val_videos)}")
    print(f" - Test:  {len(test_videos)}")

if __name__ == "__main__":
    create_splits()