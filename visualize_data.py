"""
Visualize samples from preprocessed train.npz
----------------------------------------------
Shows per-token time series features for selected samples.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

DEFAULT_FEATURE_NAMES = [
    'vert_aperture',
    'outer_vert_aperture',
    'horiz_spread',
    'inner_area',
    'outer_area',
    'compactness',
    'lip_speed',
    'rms_energy',
]


def prettify_feature_name(name):
    return str(name).replace('_', ' ').title()


def get_feature_names(data, n_features):
    if 'feature_names' in data:
        names = [str(n) for n in data['feature_names'].tolist()]
        if len(names) >= n_features:
            return [prettify_feature_name(n) for n in names[:n_features]]

    return [prettify_feature_name(n) for n in DEFAULT_FEATURE_NAMES[:n_features]]


def get_plot_feature_selection(data, n_features, exclude_names=None):
    """Return (indices, pretty_names) for features to visualize."""
    if exclude_names is None:
        exclude_names = set()
    exclude_names = {str(n).strip().lower() for n in exclude_names}

    if 'feature_names' in data:
        raw_names = [str(n) for n in data['feature_names'].tolist()[:n_features]]
    else:
        raw_names = DEFAULT_FEATURE_NAMES[:n_features]

    indices = [i for i, name in enumerate(raw_names) if str(name).strip().lower() not in exclude_names]
    names = [prettify_feature_name(raw_names[i]) for i in indices]
    return indices, names


def plot_sample(
    full_features,
    token_segments,
    token_sequence,
    speaker,
    video_id,
    fps,
    feature_names,
    token_label='Tokens',
    ax_row=None,
    fig=None,
):
    """Plot full video features + per-token segment overlay for one sample."""
    n_features = full_features.shape[1]
    T = full_features.shape[0]
    time_axis = np.arange(T) / fps

    if ax_row is None:
        fig, ax_row = plt.subplots(n_features, 1, figsize=(14, 10), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(token_sequence)))

    # Find true frame offsets for each segment by matching against full_features
    seg_offsets = []
    for seg in token_segments:
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

        # Overlay per-token segments at their true positions
        for seg_idx, (seg, token) in enumerate(zip(token_segments, token_sequence)):
            t = (seg_offsets[seg_idx] + np.arange(len(seg))) / fps
            ax.plot(t, seg[:, f_idx], color=colors[seg_idx], linewidth=1.5,
                    label=f"'{token}'" if f_idx == 0 else None)

        ax.set_ylabel(feature_names[f_idx], fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    ax_row[-1].set_xlabel('Time (s)', fontsize=10)
    ax_row[0].set_title(f'Speaker {speaker} — Video {video_id} — {token_label}: {" ".join(map(str, token_sequence))}',
                        fontsize=11, fontweight='bold')
    if ax_row[0].get_legend() is None:
        ax_row[0].legend(loc='upper right', fontsize=7, ncol=min(8, len(token_sequence)))


def plot_token_segments_grid(token_segments, token_sequence, speaker, video_id, fps, feature_names, token_label='Token'):
    """Plot each token segment as a separate subplot column."""
    n_tokens = len(token_sequence)
    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, n_tokens, figsize=(2.2 * n_tokens, max(6, 1.2 * n_features)),
                             sharex=False, sharey='row')

    if n_features == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_tokens == 1:
        axes = np.expand_dims(axes, axis=1)

    for d_idx in range(n_tokens):
        seg = token_segments[d_idx]
        t = np.arange(len(seg)) / fps
        for f_idx in range(n_features):
            ax = axes[f_idx, d_idx]
            ax.plot(t, seg[:, f_idx], color=plt.cm.tab10(d_idx / 10), linewidth=1.2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)
            if d_idx == 0:
                ax.set_ylabel(feature_names[f_idx], fontsize=8)
            if f_idx == 0:
                ax.set_title(f"'{token_sequence[d_idx]}'", fontsize=10, fontweight='bold')
            if f_idx == n_features - 1:
                ax.set_xlabel('s', fontsize=7)

    fig.suptitle(f'Per-{token_label} Segments — Speaker {speaker} — {video_id}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_token_comparison(data, token, feature_names, feature_indices=None, max_samples=10, token_label='token'):
    """Compare the same token across different videos/speakers."""
    matches = []
    for i in range(len(data['digit_sequences'])):
        seq = data['digit_sequences'][i]
        for d_idx, d in enumerate(seq):
            if str(d) == str(token):
                matches.append((i, d_idx, data['speakers'][i]))
                if len(matches) >= max_samples:
                    break
        if len(matches) >= max_samples:
            break

    if not matches:
        print(f"No segments found for {token_label} '{token}'")
        return None

    n_features = len(feature_names)
    if feature_indices is None:
        feature_indices = list(range(n_features))
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 8), sharex=False)
    if n_features == 1:
        axes = np.array([axes])

    for m_idx, (vid_idx, d_idx, speaker) in enumerate(matches):
        seg = data['digit_segments'][vid_idx][d_idx]
        fps = data['fps'][vid_idx]
        t = np.arange(len(seg)) / fps
        color = plt.cm.tab10(m_idx / max_samples)
        for f_idx in range(n_features):
            src_idx = feature_indices[f_idx]
            axes[f_idx].plot(t, seg[:, src_idx], color=color, alpha=0.7, linewidth=1,
                             label=f'spk {speaker}' if f_idx == 0 else None)
            axes[f_idx].set_ylabel(feature_names[f_idx], fontsize=9)
            axes[f_idx].grid(True, alpha=0.3)
            axes[f_idx].tick_params(labelsize=8)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(f"{token_label.title()} '{token}' — {len(matches)} samples across speakers",
                      fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=7, ncol=min(5, len(matches)))
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize preprocessed lip features')
    parser.add_argument('--dataset', choices=['digit', 'grid'], default='digit',
                        help='Dataset to visualize from processed_data/<dataset>')
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--n_samples', type=int, default=3,
                        help='Number of random samples to visualize')
    parser.add_argument('--index', type=int, nargs='+', default=None,
                        help='Specific sample indices to visualize')
    parser.add_argument('--digit', type=str, default=None,
                        help='Compare a specific token across samples (name kept for compatibility)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', action='store_true', help='Save plots instead of showing')
    args = parser.parse_args()

    processed_dir = os.path.join('processed_data', args.dataset)
    npz_path = os.path.join(processed_dir, f'{args.split}.npz')
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found. Run preprocess.py --dataset {args.dataset} first.")
        exit(1)

    data = np.load(npz_path, allow_pickle=True)
    n_features = data['full_features'][0].shape[1]
    feature_indices, feature_names = get_plot_feature_selection(
        data,
        n_features,
        exclude_names={'rms_energy'},
    )
    if not feature_indices:
        print('ERROR: no features left to plot after exclusions.')
        exit(1)

    plot_n_features = len(feature_names)
    token_label = 'Digit' if args.dataset == 'digit' else 'Token'
    n_videos = len(data['digit_segments'])
    print(
        f"Loaded {args.dataset}/{args.split} set: {n_videos} videos "
        f"({plot_n_features}/{n_features} features plotted; excluded: rms_energy)"
    )

    out_dir = os.path.join('plots', args.dataset)
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

        print(f"\nSample {idx}: speaker={speaker}, tokens={' '.join(str(d) for d in digits)}, "
              f"frames={full_feat.shape[0]}, fps={fps}")

        # Plot 1: Full video with token segment overlay
        full_feat_plot = full_feat[:, feature_indices]
        segs_plot = [seg[:, feature_indices] for seg in segs]

        fig, axes = plt.subplots(plot_n_features, 1, figsize=(14, max(8, 1.6 * plot_n_features)), sharex=True)
        if plot_n_features == 1:
            axes = np.array([axes])
        plot_sample(
            full_feat_plot,
            segs_plot,
            digits,
            speaker,
            vid_id,
            fps,
            feature_names=feature_names,
            token_label=token_label,
            ax_row=axes,
            fig=fig,
        )
        fig.tight_layout()
        if args.save:
            fig.savefig(os.path.join(out_dir, f'{args.split}_sample{idx}_overview.png'), dpi=150)
            print(f"  Saved overview plot")
        else:
            plt.show()
        plt.close(fig)

        # Plot 2: Per-token grid
        fig2 = plot_token_segments_grid(
            segs_plot,
            digits,
            speaker,
            vid_id,
            fps,
            feature_names=feature_names,
            token_label=token_label,
        )
        if args.save:
            fig2.savefig(os.path.join(out_dir, f'{args.split}_sample{idx}_tokens.png'), dpi=150)
            print(f"  Saved token grid plot")
        else:
            plt.show()
        plt.close(fig2)

    # --- Cross-sample token comparison ---
    if args.digit is not None:
        print(f"\nComparing {token_label.lower()} '{args.digit}' across samples...")
        fig3 = plot_token_comparison(
            data,
            args.digit,
            feature_names=feature_names,
            feature_indices=feature_indices,
            token_label=token_label.lower(),
        )
        if fig3:
            if args.save:
                fig3.savefig(os.path.join(out_dir, f'{args.split}_{args.digit}_compare.png'), dpi=150)
                print(f"  Saved token comparison plot")
            else:
                plt.show()
            plt.close(fig3)
