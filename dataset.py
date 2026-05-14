import numpy as np
import torch
from torch.utils.data import Dataset

from model import CHAR_TO_IDX, N_CLASSES


MAX_SEQ_LEN = 30
NEG_RATIO = 1



def char_to_idx(c):
    """Convert a digit character (0-9 or !) to an integer index."""
    return CHAR_TO_IDX[c]


def _infer_n_features(digit_segments):
    for video_segments in digit_segments:
        for seg in video_segments:
            if seg is not None and len(seg) > 0:
                return seg.shape[1]
    raise ValueError('Could not infer feature dimension from dataset')


# --- Dataset ---
class LipVerificationDataset(Dataset):
    """
    Per-digit verification dataset.
    Each sample is a (lip_segment, claimed_digit_idx, match_label) triplet.
    """

    def __init__(self, npz_path, max_seq_len=MAX_SEQ_LEN, neg_ratio=NEG_RATIO, seed=42):
        data = np.load(npz_path, allow_pickle=True)
        self.digit_segments = data['digit_segments']
        self.digit_sequences = data['digit_sequences']
        self.max_seq_len = max_seq_len
        self.rng = np.random.RandomState(seed)
        self.n_features = _infer_n_features(self.digit_segments)

        # Build flat list of (segment_features, char_idx)
        self.segments = []
        for vid_idx in range(len(self.digit_segments)):
            segs = self.digit_segments[vid_idx]
            digits = self.digit_sequences[vid_idx]
            for seg, digit in zip(segs, digits):
                self.segments.append((seg, char_to_idx(str(digit))))

        # Build pairs: positive + negative
        all_indices = list(range(N_CLASSES))
        self.pairs = []
        for i, (_, cidx) in enumerate(self.segments):
            self.pairs.append((i, cidx, 1))
            wrong = [d for d in all_indices if d != cidx]
            for _ in range(neg_ratio):
                self.pairs.append((i, self.rng.choice(wrong), 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seg_idx, claimed, label = self.pairs[idx]
        seg_features, _ = self.segments[seg_idx]

        t = seg_features.shape[0]
        if t >= self.max_seq_len:
            feat = seg_features[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            feat = np.zeros((self.max_seq_len, self.n_features), dtype=np.float32)
            feat[:t] = seg_features
            mask = np.zeros(self.max_seq_len, dtype=np.float32)
            mask[:t] = 1.0

        return (
            torch.FloatTensor(feat),
            torch.FloatTensor(mask),
            torch.LongTensor([claimed]),
            torch.FloatTensor([label]),
        )

class SequenceVerificationDataset(Dataset):
    """
    Full-sequence verification: verify entire 8-digit sequence at once.
    """

    def __init__(self, npz_path, max_seg_len=MAX_SEQ_LEN, neg_ratio=NEG_RATIO, seed=42):
        data = np.load(npz_path, allow_pickle=True)
        self.digit_segments = data['digit_segments']
        self.digit_sequences = data['digit_sequences']
        self.max_seg_len = max_seg_len
        self.rng = np.random.RandomState(seed)
        self.n_videos = len(self.digit_segments)
        self.n_features = _infer_n_features(self.digit_segments)

        self.pairs = []
        for i in range(self.n_videos):
            digits = [char_to_idx(str(d)) for d in self.digit_sequences[i]]
            self.pairs.append((i, digits, 1))
            for _ in range(neg_ratio):
                wrong_digits = digits.copy()
                if self.rng.random() < 0.5:
                    self.rng.shuffle(wrong_digits)
                    if wrong_digits == digits:
                        wrong_digits[0] = (wrong_digits[0] + 1) % N_CLASSES
                else:
                    n_replace = self.rng.randint(2, min(5, len(wrong_digits)) + 1)
                    positions = self.rng.choice(len(wrong_digits), n_replace, replace=False)
                    for pos in positions:
                        wrong_digits[pos] = self.rng.choice(
                            [d for d in range(N_CLASSES) if d != wrong_digits[pos]]
                        )
                self.pairs.append((i, wrong_digits, 0))

    def __len__(self):
        return len(self.pairs)

    def _pad_segment(self, seg):
        t = seg.shape[0]
        if t >= self.max_seg_len:
            feat = seg[:self.max_seg_len].astype(np.float32)
            mask = np.ones(self.max_seg_len, dtype=np.float32)
        else:
            feat = np.zeros((self.max_seg_len, self.n_features), dtype=np.float32)
            feat[:t] = seg
            mask = np.zeros(self.max_seg_len, dtype=np.float32)
            mask[:t] = 1.0
        return feat, mask

    def __getitem__(self, idx):
        vid_idx, claimed_digits, label = self.pairs[idx]
        segments = self.digit_segments[vid_idx]

        all_feats, all_masks = [], []
        for seg in segments:
            f, m = self._pad_segment(seg)
            all_feats.append(f)
            all_masks.append(m)

        n_digits = len(segments)
        return (
            torch.FloatTensor(np.array(all_feats)),
            torch.FloatTensor(np.array(all_masks)),
            torch.LongTensor(claimed_digits),
            torch.FloatTensor([label]),
            n_digits,
        )


class LipTranscriptionDataset(Dataset):
    """
    Seq2seq transcription dataset.
    Each sample is a full utterance represented as a sequence of digit segments.
    """

    def __init__(self, npz_path, max_seg_len=MAX_SEQ_LEN):
        data = np.load(npz_path, allow_pickle=True)
        self.digit_segments = data['digit_segments']
        self.digit_sequences = data['digit_sequences']
        self.max_seg_len = max_seg_len
        self.n_features = _infer_n_features(self.digit_segments)

    def __len__(self):
        return len(self.digit_segments)

    def _pad_segment(self, seg):
        t = seg.shape[0]
        if t >= self.max_seg_len:
            feat = seg[:self.max_seg_len].astype(np.float32)
            mask = np.ones(self.max_seg_len, dtype=np.float32)
        else:
            feat = np.zeros((self.max_seg_len, self.n_features), dtype=np.float32)
            feat[:t] = seg
            mask = np.zeros(self.max_seg_len, dtype=np.float32)
            mask[:t] = 1.0
        return feat, mask

    def __getitem__(self, idx):
        segments = self.digit_segments[idx]
        digits = [CHAR_TO_IDX[str(d)] for d in self.digit_sequences[idx]]

        all_feats, all_masks = [], []
        for seg in segments:
            f, m = self._pad_segment(seg)
            all_feats.append(f)
            all_masks.append(m)

        n_digits = len(segments)
        return (
            torch.FloatTensor(np.array(all_feats)),
            torch.FloatTensor(np.array(all_masks)),
            torch.LongTensor(digits),
            n_digits,
        )
