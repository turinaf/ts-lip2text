import json
from pathlib import Path
import re
import numpy as np
import pandas as pd


def _find_segment_ranges(video_feats: np.ndarray, segments: np.ndarray) -> list[list[int | None]]:
    """Return (start, end) frame indices in full video for each digit segment."""
    ranges: list[list[int | None]] = []
    start_search = 0

    for seg in segments:
        seg_arr = np.asarray(seg)
        seg_len = int(seg_arr.shape[0])
        found_start = None

        if seg_len > 0 and seg_len <= video_feats.shape[0]:
            max_start = video_feats.shape[0] - seg_len
            for s in range(start_search, max_start + 1):
                candidate = video_feats[s : s + seg_len]
                if candidate.shape == seg_arr.shape and np.allclose(candidate, seg_arr, rtol=1e-6, atol=1e-8):
                    found_start = s
                    break

        if found_start is None:
            ranges.append([None, None])
            continue

        found_end = found_start + seg_len - 1
        ranges.append([int(found_start), int(found_end)])
        start_search = found_end + 1

    return ranges


def _extract_filename_sequence(video_id: str) -> list[str] | None:
    """Extract 8-symbol digit sequence token from filename embedded in video_id."""
    filename = str(video_id).split("/")[-1]
    for part in filename.split("_"):
        if re.fullmatch(r"[0-9!]{8}", part):
            return list(part)
    return None


def convert_npz_to_csv(npz_path: Path, out_csv: Path) -> None:
    data = np.load(npz_path, allow_pickle=True)
    feature_names = [str(x) for x in data["feature_names"]]
    video_ids = data["video_ids"]
    full_features = data["full_features"]
    digit_sequences = data["digit_sequences"]
    digit_segments = data["digit_segments"]

    rows = []
    frame_rows = []
    for i, vid in enumerate(video_ids):
        video_id = str(vid)
        feats = np.asarray(full_features[i])
        # serialize each feature (column) as JSON list of floats for that video
        feature_dict = {}
        # feats shape: (frames, num_features)
        for j, name in enumerate(feature_names):
            col_vals = feats[:, j].astype(float).tolist() if feats.size else []
            feature_dict[name] = json.dumps(col_vals)

        digits = digit_sequences[i]
        digits_list = digits.tolist() if hasattr(digits, "tolist") else list(digits)
        segment_ranges = _find_segment_ranges(feats, digit_segments[i])
        filename_digits = _extract_filename_sequence(video_id)

        row = {
            "video_id": video_id,
            **feature_dict,
            "digit_sequences": json.dumps([str(x) for x in digits_list]),
            "digit_sequences_filename": json.dumps(filename_digits) if filename_digits is not None else "",
            "digit_segments": json.dumps(segment_ranges),
        }
        rows.append(row)

        # frame-wise numeric export (no stringified feature vectors)
        for frame_idx, frame_values in enumerate(feats):
            frame_row = {
                "video_id": video_id,
                "frame_idx": int(frame_idx),
            }
            for j, name in enumerate(feature_names):
                frame_row[name] = float(frame_values[j])
            frame_rows.append(frame_row)

    df = pd.DataFrame(rows)
    frame_df = pd.DataFrame(frame_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_frames_csv = out_csv.with_name(f"{out_csv.stem}_frames.csv")
    frame_df.to_csv(out_frames_csv, index=False)


def main():
    base = Path("processed_data/grid")
    for name in ("train", "test"):
        npz = base / f"{name}.npz"
        out = Path("csv/grid") / f"{name}.csv"
        if not npz.exists():
            print(f"Skipping missing {npz}")
            continue
        print(f"Converting {npz} -> {out}")
        convert_npz_to_csv(npz, out)


if __name__ == "__main__":
    main()
