import numpy as np
import torch
from torch.utils.data import Dataset

from model import CHAR_TO_IDX, N_CLASSES, VOCAB


MAX_SEQ_LEN = 30
NEG_RATIO = 1
EXPECTED_N_DIGITS = 8



def char_to_idx(c):
    """Convert a digit character (0-9 or !) to an integer index."""
    return CHAR_TO_IDX[c]


def _as_token_list(sequence):
    if hasattr(sequence, 'tolist'):
        sequence = sequence.tolist()
    return [str(token) for token in sequence]


def _build_token_to_idx(digit_sequences, dataset='digit'):
    if dataset == 'digit':
        return {token: idx for idx, token in enumerate(VOCAB)}

    vocab = sorted({token for seq in digit_sequences for token in _as_token_list(seq)})
    return {token: idx for idx, token in enumerate(vocab)}


def _encode_sequence(sequence, token_to_idx):
    tokens = _as_token_list(sequence)
    try:
        return [token_to_idx[token] for token in tokens]
    except KeyError as exc:
        raise KeyError(f"Unknown token '{exc.args[0]}' found while encoding sequence") from exc


def _infer_n_features(digit_segments):
    for video_segments in digit_segments:
        for seg in video_segments:
            if seg is not None and len(seg) > 0:
                return seg.shape[1]
    raise ValueError('Could not infer feature dimension from dataset')


def _filter_fixed_length_samples(digit_segments, digit_sequences, expected_len=EXPECTED_N_DIGITS):
    """Keep only samples where both sequence and segment count match expected length."""
    keep_idx = []
    for i, (segs, seq) in enumerate(zip(digit_segments, digit_sequences)):
        if len(segs) == expected_len and len(seq) == expected_len:
            keep_idx.append(i)

    if not keep_idx:
        raise ValueError(
            f'No samples with exactly {expected_len} digits found after filtering'
        )

    return digit_segments[keep_idx], digit_sequences[keep_idx]


def _prepare_samples(npz_path, dataset='digit', expected_len=EXPECTED_N_DIGITS):
    data = np.load(npz_path, allow_pickle=True)
    digit_segments = data['digit_segments']
    digit_sequences = data['digit_sequences']

    token_to_idx = _build_token_to_idx(digit_sequences, dataset=dataset)
    return digit_segments, digit_sequences, token_to_idx


# --- Dataset ---
class LipVerificationDataset(Dataset):
    """
    Per-digit verification dataset.
    Each sample is a (lip_segment, claimed_digit_idx, match_label) triplet.
    """

    def __init__(self, npz_path, dataset='digit', token_to_idx=None,
                 max_seq_len=MAX_SEQ_LEN, neg_ratio=NEG_RATIO, seed=42,
                 expected_len=EXPECTED_N_DIGITS):
        self.dataset = dataset
        self.digit_segments, self.digit_sequences, inferred_token_to_idx = _prepare_samples(
            npz_path,
            dataset=dataset,
            expected_len=expected_len,
        )
        self.token_to_idx = token_to_idx or inferred_token_to_idx
        self.max_seq_len = max_seq_len
        self.rng = np.random.RandomState(seed)
        self.n_features = _infer_n_features(self.digit_segments)
        self.vocab_size = len(self.token_to_idx)

        # Build flat list of (segment_features, char_idx)
        self.segments = []
        for vid_idx in range(len(self.digit_segments)):
            segs = self.digit_segments[vid_idx]
            digits = self.digit_sequences[vid_idx]
            for seg, digit in zip(segs, digits):
                self.segments.append((seg, self.token_to_idx[str(digit)]))

        # Build pairs: positive + negative
        all_indices = list(range(self.vocab_size))
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

    def __init__(self, npz_path, dataset='digit', token_to_idx=None,
                 max_seg_len=MAX_SEQ_LEN, neg_ratio=NEG_RATIO, seed=42,
                 expected_len=EXPECTED_N_DIGITS):
        self.dataset = dataset
        self.digit_segments, self.digit_sequences, inferred_token_to_idx = _prepare_samples(
            npz_path,
            dataset=dataset,
            expected_len=expected_len,
        )
        self.token_to_idx = token_to_idx or inferred_token_to_idx
        self.max_seg_len = max_seg_len
        self.rng = np.random.RandomState(seed)
        self.n_videos = len(self.digit_segments)
        self.n_features = _infer_n_features(self.digit_segments)
        self.vocab_size = len(self.token_to_idx)

        self.pairs = []
        for i in range(self.n_videos):
            digits = _encode_sequence(self.digit_sequences[i], self.token_to_idx)
            self.pairs.append((i, digits, 1))
            for _ in range(neg_ratio):
                wrong_digits = digits.copy()
                if self.rng.random() < 0.5:
                    self.rng.shuffle(wrong_digits)
                    if wrong_digits == digits:
                        wrong_digits[0] = (wrong_digits[0] + 1) % self.vocab_size
                else:
                    max_replace = min(5, len(wrong_digits))
                    n_replace = 1 if max_replace < 2 else self.rng.randint(2, max_replace + 1)
                    positions = self.rng.choice(len(wrong_digits), n_replace, replace=False)
                    for pos in positions:
                        wrong_digits[pos] = self.rng.choice(
                            [d for d in range(self.vocab_size) if d != wrong_digits[pos]]
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

    def __init__(self, npz_path, dataset='digit', token_to_idx=None,
                 max_seg_len=MAX_SEQ_LEN, expected_len=EXPECTED_N_DIGITS):
        self.dataset = dataset
        self.digit_segments, self.digit_sequences, inferred_token_to_idx = _prepare_samples(
            npz_path,
            dataset=dataset,
            expected_len=expected_len,
        )
        self.token_to_idx = token_to_idx or inferred_token_to_idx
        self.max_seg_len = max_seg_len
        self.n_features = _infer_n_features(self.digit_segments)
        self.vocab_size = len(self.token_to_idx)

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
        digits = _encode_sequence(self.digit_sequences[idx], self.token_to_idx)

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
