"""Print summary information for preprocessed datasets and example pairs.

Usage:
	python dataset_info.py
"""

from pathlib import Path

import numpy as np

from model import VOCAB
from train import LipVerificationDataset, SequenceVerificationDataset


def print_npz_info(path: Path) -> None:
	data = np.load(path, allow_pickle=True)
	print(f"\n=== {path} ===")
	print("keys:", data.files)
	for key in data.files:
		arr = data[key]
		print(f"- {key}: shape={arr.shape}, dtype={arr.dtype}")


def print_digit_pair_examples(train_path: Path, seed: int = 42) -> None:
	print("\n=== One Positive/Negative Pair (Digit Verification) ===")
	ds = LipVerificationDataset(str(train_path), seed=seed)

	pos_idx = next(i for i, (_, _, y) in enumerate(ds.pairs) if y == 1)
	neg_idx = next(i for i, (_, _, y) in enumerate(ds.pairs) if y == 0)

	for name, idx in [("Positive", pos_idx), ("Negative", neg_idx)]:
		seg_idx, claimed, label = ds.pairs[idx]
		seg_feat, true_digit = ds.segments[seg_idx]
		flat = seg_feat.reshape(-1)

		print(f"\n{name} pair idx={idx}, label={label}")
		print(f"- true_digit={VOCAB[true_digit]} ({true_digit})")
		print(f"- claimed_digit={VOCAB[claimed]} ({claimed})")
		print(f"- segment_shape={seg_feat.shape}, dtype={seg_feat.dtype}")
		print(f"- first6_features={flat[:6]}")


def print_sequence_pair_examples(train_path: Path, seed: int = 42) -> None:
	print("\n=== One Positive/Negative Pair (Sequence Verification) ===")
	ds = SequenceVerificationDataset(str(train_path), seed=seed)

	pos_idx = next(i for i, (_, _, y) in enumerate(ds.pairs) if y == 1)
	neg_idx = next(i for i, (_, _, y) in enumerate(ds.pairs) if y == 0)

	for name, idx in [("Positive", pos_idx), ("Negative", neg_idx)]:
		vid_idx, claimed_digits, label = ds.pairs[idx]
		true_digits = [int(x) for x in ds.digit_sequences[vid_idx]]

		print(f"\n{name} pair idx={idx}, label={label}")
		print(f"- video_idx={vid_idx}")
		print(f"- true_sequence={''.join(map(str, true_digits))}")
		print(f"- claimed_sequence={''.join(str(VOCAB[d]) for d in claimed_digits)}")
		print(f"- num_segments={len(ds.digit_segments[vid_idx])}")


def main() -> None:
	train_path = Path("processed_data/train.npz")
	test_path = Path("processed_data/test.npz")

	if not train_path.exists() or not test_path.exists():
		raise FileNotFoundError("Expected files: processed_data/train.npz and processed_data/test.npz")

	print_npz_info(train_path)
	print_npz_info(test_path)
	print_digit_pair_examples(train_path, seed=42)
	print_sequence_pair_examples(train_path, seed=42)


if __name__ == "__main__":
	main()
