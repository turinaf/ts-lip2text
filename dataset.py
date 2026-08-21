import numpy as np
import torch
from torch.utils.data import Dataset
import argparse

from model import CHAR_TO_IDX, N_CLASSES, VOCAB
from transforms import compute_feature_stats, correct_lip_speed_fps, resample_segment, standardize_segment


MAX_SEQ_LEN = 30
NEG_RATIO = 1
EXPECTED_N_DIGITS = 8
RMS_FEATURE_NAME = 'rms_energy'
SEG_LEN = 16


def _feature_name_to_index(feature_names):
    return {name: idx for idx, name in enumerate(feature_names or [])}



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


def _adapt_feature_dim(seg, target_dim):
    """Slice or zero-pad a segment to the dataset/model feature width."""
    seg = np.asarray(seg, dtype=np.float32)
    current_dim = seg.shape[1]
    if current_dim == target_dim:
        return seg
    if current_dim > target_dim:
        return seg[:, :target_dim]

    padded = np.zeros((seg.shape[0], target_dim), dtype=np.float32)
    padded[:, :current_dim] = seg
    return padded


def _filter_fixed_length_samples(digit_segments, digit_sequences, fps, expected_len=EXPECTED_N_DIGITS):
    """Keep only samples where both sequence and segment count match expected length."""
    keep_idx = []
    for i, (segs, seq) in enumerate(zip(digit_segments, digit_sequences)):
        if len(segs) == expected_len and len(seq) == expected_len:
            keep_idx.append(i)

    if not keep_idx:
        raise ValueError(
            f'No samples with exactly {expected_len} digits found after filtering'
        )

    return digit_segments[keep_idx], digit_sequences[keep_idx], [fps[i] for i in keep_idx]


def _prepare_samples(npz_path, dataset='digit', expected_len=EXPECTED_N_DIGITS, speaker_filter=None):
    data = np.load(npz_path, allow_pickle=True)
    digit_segments = data['digit_segments']
    digit_sequences = data['digit_sequences']
    feature_names = [str(name) for name in data['feature_names'].tolist()] if 'feature_names' in data else None
    fps = data['fps'].tolist() if 'fps' in data.files else [25.0] * len(digit_segments)
    speakers = [str(s) for s in data['speakers']] if 'speakers' in data.files else None

    if speaker_filter is not None:
        if speakers is None:
            raise ValueError('speaker_filter requested but npz has no speakers array')
        keep = [i for i, spk in enumerate(speakers) if spk in speaker_filter]
        digit_segments = digit_segments[keep]
        digit_sequences = digit_sequences[keep]
        fps = [fps[i] for i in keep]

    # Digit dataset is expected to contain fixed-length sequences (default: 8).
    # GRID uses variable-length word sequences, so we keep all samples.
    if dataset == 'digit':
        digit_segments, digit_sequences, fps = _filter_fixed_length_samples(
            digit_segments,
            digit_sequences,
            fps,
            expected_len=expected_len,
        )

    token_to_idx = _build_token_to_idx(digit_sequences, dataset=dataset)
    return {
        'segments': digit_segments,
        'sequences': digit_sequences,
        'token_to_idx': token_to_idx,
        'feature_names': feature_names,
        'fps': fps,
    }


def _transform_segment(seg, fps, n_features, resample, seg_len, feature_stats):
    seg = _adapt_feature_dim(seg, n_features)
    if resample:
        seg = correct_lip_speed_fps(seg, fps)
        seg = resample_segment(seg, seg_len)
    if feature_stats is not None:
        seg = standardize_segment(seg, feature_stats['mean'], feature_stats['std'])
    return seg


def compute_split_stats(npz_path, dataset='digit', expected_len=EXPECTED_N_DIGITS,
                        resample=True, seg_len=SEG_LEN, speaker_filter=None):
    """Per-feature stats over transformed (unstandardized) segments of a split."""
    prepared = _prepare_samples(
        npz_path, dataset=dataset, expected_len=expected_len, speaker_filter=speaker_filter,
    )
    segments = prepared['segments']
    fps = prepared['fps']
    feature_names = prepared['feature_names']
    n_features = len(feature_names) if feature_names else _infer_n_features(segments)

    def iter_segments():
        for vid_idx in range(len(segments)):
            video_fps = fps[vid_idx]
            for seg in segments[vid_idx]:
                yield _transform_segment(seg, video_fps, n_features, resample, seg_len, None)

    return compute_feature_stats(iter_segments(), n_features)


# --- Dataset ---
class LipVerificationDataset(Dataset):
    """
    Per-digit verification dataset.
    Each sample is a (lip_segment, claimed_digit_idx, match_label) triplet.
    """

    def __init__(self, npz_path, dataset='digit', token_to_idx=None,
                 max_seq_len=MAX_SEQ_LEN, neg_ratio=NEG_RATIO, seed=42,
                 expected_len=EXPECTED_N_DIGITS, resample=True, seg_len=SEG_LEN,
                 feature_stats=None, speaker_filter=None):
        self.dataset = dataset
        prepared = _prepare_samples(
            npz_path, dataset=dataset, expected_len=expected_len, speaker_filter=speaker_filter,
        )
        self.digit_segments = prepared['segments']
        self.digit_sequences = prepared['sequences']
        self.token_to_idx = token_to_idx or prepared['token_to_idx']
        self.feature_names = prepared['feature_names']
        self.feature_name_to_index = _feature_name_to_index(self.feature_names)
        self.fps = prepared['fps']
        self.max_seq_len = max_seq_len
        self.pad_len = seg_len if resample else max_seq_len
        self.resample = resample
        self.seg_len = seg_len
        self.feature_stats = feature_stats
        self.rng = np.random.RandomState(seed)
        self.n_features = len(self.feature_names) if self.feature_names else _infer_n_features(self.digit_segments)
        self.vocab_size = len(self.token_to_idx)

        # Build flat list of (segment_features, char_idx, fps)
        self.segments = []
        for vid_idx in range(len(self.digit_segments)):
            segs = self.digit_segments[vid_idx]
            digits = self.digit_sequences[vid_idx]
            video_fps = self.fps[vid_idx]
            for seg, digit in zip(segs, digits):
                self.segments.append((seg, self.token_to_idx[str(digit)], video_fps))

        all_indices = list(range(self.vocab_size))
        self.pairs = []
        for i, (_, cidx, _) in enumerate(self.segments):
            self.pairs.append((i, cidx, 1))
            wrong = [d for d in all_indices if d != cidx]
            for _ in range(neg_ratio):
                self.pairs.append((i, self.rng.choice(wrong), 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seg_idx, claimed, label = self.pairs[idx]
        seg_features, _, fps = self.segments[seg_idx]
        seg_features = _transform_segment(
            seg_features, fps, self.n_features, self.resample, self.seg_len, self.feature_stats
        )

        t = seg_features.shape[0]
        if t >= self.pad_len:
            feat = seg_features[:self.pad_len].astype(np.float32)
            mask = np.ones(self.pad_len, dtype=np.float32)
        else:
            feat = np.zeros((self.pad_len, self.n_features), dtype=np.float32)
            feat[:t] = seg_features
            mask = np.zeros(self.pad_len, dtype=np.float32)
            mask[:t] = 1.0

        return (
            torch.FloatTensor(feat),
            torch.FloatTensor(mask),
            torch.LongTensor([claimed]),
            torch.FloatTensor([label]),
        )

    def get_pair_info(self, idx):
        """Return (seg_idx, true_digit, claimed_digit, label) for analysis."""
        seg_idx, claimed, label = self.pairs[idx]
        _, true_digit, _ = self.segments[seg_idx]
        return seg_idx, true_digit, claimed, label

class SequenceVerificationDataset(Dataset):
    """
    Full-sequence verification: verify an entire token sequence at once.
    """

    def __init__(self, npz_path, dataset='digit', token_to_idx=None,
                 max_seg_len=MAX_SEQ_LEN, neg_ratio=NEG_RATIO, seed=42,
                 expected_len=EXPECTED_N_DIGITS, resample=True, seg_len=SEG_LEN,
                 feature_stats=None, speaker_filter=None):
        self.dataset = dataset
        prepared = _prepare_samples(
            npz_path, dataset=dataset, expected_len=expected_len, speaker_filter=speaker_filter,
        )
        self.digit_segments = prepared['segments']
        self.digit_sequences = prepared['sequences']
        self.token_to_idx = token_to_idx or prepared['token_to_idx']
        self.feature_names = prepared['feature_names']
        self.feature_name_to_index = _feature_name_to_index(self.feature_names)
        self.fps = prepared['fps']
        self.max_seg_len = max_seg_len
        self.pad_len = seg_len if resample else max_seg_len
        self.resample = resample
        self.seg_len = seg_len
        self.feature_stats = feature_stats
        self.rng = np.random.RandomState(seed)
        self.n_videos = len(self.digit_segments)
        self.n_features = len(self.feature_names) if self.feature_names else _infer_n_features(self.digit_segments)
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

    def _pad_segment(self, seg, fps):
        seg = _transform_segment(
            seg, fps, self.n_features, self.resample, self.seg_len, self.feature_stats
        )
        t = seg.shape[0]
        if t >= self.pad_len:
            feat = seg[:self.pad_len].astype(np.float32)
            mask = np.ones(self.pad_len, dtype=np.float32)
        else:
            feat = np.zeros((self.pad_len, self.n_features), dtype=np.float32)
            feat[:t] = seg
            mask = np.zeros(self.pad_len, dtype=np.float32)
            mask[:t] = 1.0
        return feat, mask

    def __getitem__(self, idx):
        vid_idx, claimed_digits, label = self.pairs[idx]
        segments = self.digit_segments[vid_idx]

        all_feats, all_masks = [], []
        for seg, fps in zip(segments, [self.fps[vid_idx]] * len(segments)):
            f, m = self._pad_segment(seg, fps)
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
                 max_seg_len=MAX_SEQ_LEN, expected_len=EXPECTED_N_DIGITS,
                 resample=True, seg_len=SEG_LEN, feature_stats=None, speaker_filter=None):
        self.dataset = dataset
        prepared = _prepare_samples(
            npz_path, dataset=dataset, expected_len=expected_len, speaker_filter=speaker_filter,
        )
        self.digit_segments = prepared['segments']
        self.digit_sequences = prepared['sequences']
        self.token_to_idx = token_to_idx or prepared['token_to_idx']
        self.feature_names = prepared['feature_names']
        self.feature_name_to_index = _feature_name_to_index(self.feature_names)
        self.fps = prepared['fps']
        self.max_seg_len = max_seg_len
        self.pad_len = seg_len if resample else max_seg_len
        self.resample = resample
        self.seg_len = seg_len
        self.feature_stats = feature_stats
        self.n_features = len(self.feature_names) if self.feature_names else _infer_n_features(self.digit_segments)
        self.vocab_size = len(self.token_to_idx)

    def __len__(self):
        return len(self.digit_segments)

    def _pad_segment(self, seg, fps):
        seg = _transform_segment(
            seg, fps, self.n_features, self.resample, self.seg_len, self.feature_stats
        )
        t = seg.shape[0]
        if t >= self.pad_len:
            feat = seg[:self.pad_len].astype(np.float32)
            mask = np.ones(self.pad_len, dtype=np.float32)
        else:
            feat = np.zeros((self.pad_len, self.n_features), dtype=np.float32)
            feat[:t] = seg
            mask = np.zeros(self.pad_len, dtype=np.float32)
            mask[:t] = 1.0
        return feat, mask

    def __getitem__(self, idx):
        segments = self.digit_segments[idx]
        digits = _encode_sequence(self.digit_sequences[idx], self.token_to_idx)

        all_feats, all_masks = [], []
        for seg, fps in zip(segments, [self.fps[idx]] * len(segments)):
            f, m = self._pad_segment(seg, fps)
            all_feats.append(f)
            all_masks.append(m)

        n_digits = len(segments)
        return (
            torch.FloatTensor(np.array(all_feats)),
            torch.FloatTensor(np.array(all_masks)),
            torch.LongTensor(digits),
            n_digits,
        )


def sequence_collate_fn(batch):
    """Collate sequences with variable number of digits by padding to max in batch."""
    max_digits = max(item[4] for item in batch)
    t = batch[0][0].shape[1]
    f = batch[0][0].shape[2]

    batch_feats = []
    batch_masks = []
    batch_digits = []
    batch_labels = []
    batch_seq_masks = []

    for feats, masks, digits, label, n_dig in batch:
        pad_n = max_digits - n_dig
        if pad_n > 0:
            feats = torch.cat([feats, torch.zeros(pad_n, t, f)], dim=0)
            masks = torch.cat([masks, torch.zeros(pad_n, t)], dim=0)
            digits = torch.cat([digits, torch.zeros(pad_n, dtype=torch.long)], dim=0)
        seq_mask = torch.zeros(max_digits)
        seq_mask[:n_dig] = 1.0
        batch_feats.append(feats)
        batch_masks.append(masks)
        batch_digits.append(digits)
        batch_labels.append(label)
        batch_seq_masks.append(seq_mask)

    return (
        torch.stack(batch_feats),
        torch.stack(batch_masks),
        torch.stack(batch_digits),
        torch.stack(batch_labels),
        torch.stack(batch_seq_masks),
    )


def transcription_collate_fn(batch, pad_idx, eos_idx):
    """Pad variable-length segment and token sequences for seq2seq training."""
    max_digits = max(item[3] for item in batch)
    t = batch[0][0].shape[1]
    f = batch[0][0].shape[2]
    max_tgt_len = max_digits + 1  # digits + EOS

    batch_feats = []
    batch_masks = []
    batch_src_pad = []
    batch_targets = []

    for feats, masks, digits, n_dig in batch:
        pad_n = max_digits - n_dig
        if pad_n > 0:
            feats = torch.cat([feats, torch.zeros(pad_n, t, f)], dim=0)
            masks = torch.cat([masks, torch.zeros(pad_n, t)], dim=0)

        src_pad = torch.ones(max_digits, dtype=torch.bool)
        src_pad[:n_dig] = False

        target = torch.full((max_tgt_len,), pad_idx, dtype=torch.long)
        target[:n_dig] = digits
        target[n_dig] = eos_idx

        batch_feats.append(feats)
        batch_masks.append(masks)
        batch_src_pad.append(src_pad)
        batch_targets.append(target)

    return (
        torch.stack(batch_feats),
        torch.stack(batch_masks),
        torch.stack(batch_src_pad),
        torch.stack(batch_targets),
    )


def _print_sample_summary(ds, sample_idx, dataset_name, feature_name='lip_speed'):
    feats, masks, digits, n_tokens = ds[sample_idx]
    idx_to_token = {idx: tok for tok, idx in ds.token_to_idx.items()}
    decoded = [idx_to_token[int(x)] for x in digits[:n_tokens]]

    print(f'Sample {sample_idx}')
    print(f'- n_tokens={n_tokens}')
    print(f'- tokens={decoded}')
    print(f'- features_shape={tuple(feats.shape)}')
    print(f'- masks_shape={tuple(masks.shape)}')

    feature_idx = ds.feature_name_to_index[feature_name]
    feature_values = []
    for token_idx in range(n_tokens):
        valid_len = int(masks[token_idx].sum().item())
        if valid_len > 0:
            feature_values.extend(feats[token_idx, :valid_len, feature_idx].tolist())
    print(f'- {feature_name}_vector_len={len(feature_values)}')
    print(f'- {feature_name}_vector={feature_values}')

    if dataset_name == 'digit':
        print(f"- as_digit_string={''.join(decoded)}")
    else:
        print(f"- as_phrase={' '.join(decoded)}")


def _parse_args():
    parser = argparse.ArgumentParser(description='Inspect preprocessed lip-reading datasets.')
    parser.add_argument('--dataset', choices=['digit', 'grid'], default='digit', help='Dataset type.')
    parser.add_argument('--split', choices=['train', 'test'], default='train', help='Data split to inspect.')
    parser.add_argument('--npz-path', default='', help='Optional explicit .npz path override.')
    parser.add_argument('--samples', type=int, default=3, help='Number of samples to print.')
    parser.add_argument('--max-seg-len', type=int, default=MAX_SEQ_LEN, help='Max frames per segment for padding/truncation.')
    parser.add_argument(
        '--feature-name',
        default='lip_speed',
        help='Feature trajectory to print across the full video.',
    )
    return parser.parse_args()


def _default_npz_path(dataset_name, split):
    return f'processed_data/{dataset_name}/{split}.npz'


if __name__ == '__main__':
    args = _parse_args()
    npz_path = args.npz_path or _default_npz_path(args.dataset, args.split)

    ds = LipTranscriptionDataset(
        npz_path=npz_path,
        dataset=args.dataset,
        max_seg_len=args.max_seg_len,
    )

    if args.feature_name not in ds.feature_name_to_index:
        raise SystemExit(
            f"Unknown feature '{args.feature_name}'. Available: {sorted(ds.feature_name_to_index)}"
        )

    print(f'Dataset loaded from: {npz_path}')
    print(f'- dataset={args.dataset}')
    print(f'- split={args.split}')
    print(f'- size={len(ds)}')
    print(f'- vocab_size={ds.vocab_size}')

    n_show = min(args.samples, len(ds))
    for i in range(n_show):
        _print_sample_summary(ds, i, args.dataset, feature_name=args.feature_name)
