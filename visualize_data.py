"""
Visualize samples from preprocessed train.npz
----------------------------------------------
Shows per-digit time series features for selected samples.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

FEATURE_NAMES = ['Vert. Aperture', 'Horiz. Spread', 'Inner Lip Area', 'Compactness', 'Lip Speed']
PROCESSED_DIR = 'processed_data'


def plot_sample(full_features, digit_segments, digit_sequence, speaker, video_id, fps, ax_row=None, fig=None):
    """Plot full video features + per-digit segment overlay for one sample."""
    n_features = full_features.shape[1]
    T = full_features.shape[0]
    time_axis = np.arange(T) / fps

    if ax_row is None:
        fig, ax_row = plt.subplots(n_features, 1, figsize=(14, 10), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(digit_sequence)))

    # Find true frame offsets for each segment by matching against full_features
    seg_offsets = []
    for seg in digit_segments:
        seg_len = len(seg)
        best_offset = 0
        best_err = np.inf
        for start in range(T - seg_len + 1):
            err = np.sum((full_features[start:start+seg_len] - seg) ** 2)
            if err < best_err:
                best_err = err
                best_offset = start
                if err < 1e-12:
                    break
        seg_offsets.append(best_offset)

    for f_idx in range(n_features):
        ax = ax_row[f_idx]
        ax.plot(time_axis, full_features[:, f_idx], color='gray', alpha=0.4, linewidth=0.8)

        # Overlay per-digit segments at their true positions
        for seg_idx, (seg, digit) in enumerate(zip(digit_segments, digit_sequence)):
            t = (seg_offsets[seg_idx] + np.arange(len(seg))) / fps
            ax.plot(t, seg[:, f_idx], color=colors[seg_idx], linewidth=1.5,
                    label=f"'{digit}'" if f_idx == 0 else None)

        ax.set_ylabel(FEATURE_NAMES[f_idx], fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    ax_row[-1].set_xlabel('Time (s)', fontsize=10)
    ax_row[0].set_title(f'Speaker {speaker} — Video {video_id} — Digits: {" ".join(digit_sequence)}',
                        fontsize=11, fontweight='bold')
    if ax_row[0].get_legend() is None:
        ax_row[0].legend(loc='upper right', fontsize=7, ncol=len(digit_sequence))


def plot_digit_segments_grid(digit_segments, digit_sequence, speaker, video_id, fps):
    """Plot each digit segment as a separate subplot column."""
    n_digits = len(digit_sequence)
    n_features = 5
    fig, axes = plt.subplots(n_features, n_digits, figsize=(2.2 * n_digits, 8),
                             sharex=False, sharey='row')

    for d_idx in range(n_digits):
        seg = digit_segments[d_idx]
        t = np.arange(len(seg)) / fps
        for f_idx in range(n_features):
            ax = axes[f_idx, d_idx]
            ax.plot(t, seg[:, f_idx], color=plt.cm.tab10(d_idx / 10), linewidth=1.2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)
            if d_idx == 0:
                ax.set_ylabel(FEATURE_NAMES[f_idx], fontsize=8)
            if f_idx == 0:
                ax.set_title(f"'{digit_sequence[d_idx]}'", fontsize=10, fontweight='bold')
            if f_idx == n_features - 1:
                ax.set_xlabel('s', fontsize=7)

    fig.suptitle(f'Per-Digit Segments — Speaker {speaker} — {video_id}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_digit_comparison(data, digit, max_samples=10):
    """Compare the same digit across different videos/speakers."""
    matches = []
    for i in range(len(data['digit_sequences'])):
        seq = data['digit_sequences'][i]
        for d_idx, d in enumerate(seq):
            if str(d) == str(digit):
                matches.append((i, d_idx, data['speakers'][i]))
                if len(matches) >= max_samples:
                    break
        if len(matches) >= max_samples:
            break

    if not matches:
        print(f"No segments found for digit '{digit}'")
        return None

    n_features = 5
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 8), sharex=False)

    for m_idx, (vid_idx, d_idx, speaker) in enumerate(matches):
        seg = data['digit_segments'][vid_idx][d_idx]
        fps = data['fps'][vid_idx]
        t = np.arange(len(seg)) / fps
        color = plt.cm.tab10(m_idx / max_samples)
        for f_idx in range(n_features):
            axes[f_idx].plot(t, seg[:, f_idx], color=color, alpha=0.7, linewidth=1,
                             label=f'spk {speaker}' if f_idx == 0 else None)
            axes[f_idx].set_ylabel(FEATURE_NAMES[f_idx], fontsize=9)
            axes[f_idx].grid(True, alpha=0.3)
            axes[f_idx].tick_params(labelsize=8)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(f"Digit '{digit}' — {len(matches)} samples across speakers",
                      fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=7, ncol=min(5, len(matches)))
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize preprocessed lip features')
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--n_samples', type=int, default=3,
                        help='Number of random samples to visualize')
    parser.add_argument('--index', type=int, nargs='+', default=None,
                        help='Specific sample indices to visualize')
    parser.add_argument('--digit', type=str, default=None,
                        help='Compare a specific digit across samples')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', action='store_true', help='Save plots instead of showing')
    args = parser.parse_args()

    npz_path = os.path.join(PROCESSED_DIR, f'{args.split}.npz')
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found. Run preprocess.py first.")
        exit(1)

    data = np.load(npz_path, allow_pickle=True)
    n_videos = len(data['digit_segments'])
    print(f"Loaded {args.split} set: {n_videos} videos")

    out_dir = 'plots'
    if args.save:
        os.makedirs(out_dir, exist_ok=True)

    # Select sample indices
    if args.index is not None:
        indices = args.index
    else:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(n_videos, size=min(args.n_samples, n_videos), replace=False)

    # --- Per-sample visualizations ---
    for idx in indices:
        full_feat = data['full_features'][idx]
        segs = data['digit_segments'][idx]
        digits = data['digit_sequences'][idx]
        speaker = data['speakers'][idx]
        vid_id = data['video_ids'][idx]
        fps = data['fps'][idx]

        print(f"\nSample {idx}: speaker={speaker}, digits={' '.join(str(d) for d in digits)}, "
              f"frames={full_feat.shape[0]}, fps={fps}")

        # Plot 1: Full video with segment overlay
        fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
        plot_sample(full_feat, segs, digits, speaker, vid_id, fps, ax_row=axes, fig=fig)
        fig.tight_layout()
        if args.save:
            fig.savefig(os.path.join(out_dir, f'{args.split}_sample{idx}_overview.png'), dpi=150)
            print(f"  Saved overview plot")
        else:
            plt.show()
        plt.close(fig)

        # Plot 2: Per-digit grid
        fig2 = plot_digit_segments_grid(segs, digits, speaker, vid_id, fps)
        if args.save:
            fig2.savefig(os.path.join(out_dir, f'{args.split}_sample{idx}_digits.png'), dpi=150)
            print(f"  Saved digit grid plot")
        else:
            plt.show()
        plt.close(fig2)

    # --- Cross-sample digit comparison ---
    if args.digit is not None:
        print(f"\nComparing digit '{args.digit}' across samples...")
        fig3 = plot_digit_comparison(data, args.digit)
        if fig3:
            if args.save:
                fig3.savefig(os.path.join(out_dir, f'{args.split}_digit{args.digit}_compare.png'), dpi=150)
                print(f"  Saved digit comparison plot")
            else:
                plt.show()
            plt.close(fig3)
