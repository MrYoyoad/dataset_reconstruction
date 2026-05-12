"""Analyze the Phase 0 face-prior ablation sweep.

Loads the .pth outputs from scripts/run_phase0_face_prior_sweep.sh and produces:
  figures/phase0/face_prior/face_prior_grid.png            -- qualitative grid
  figures/phase0/face_prior/face_prior_strength_sweep.png  -- face_weight curve
  figures/phase0/face_prior/face_prior_cos_sweep.png       -- cos_weight curve
  figures/phase0/face_prior/face_prior_loss_curves.png     -- per-arm loss panels
  figures/phase0/face_prior/face_prior_landmark_evolution.png  -- winner only
  results/phase0_face_prior_sweep_<timestamp>.csv          -- metrics table

Run after all sweep arms finish:
    python -m experiments.analyze_face_prior_sweep
"""

import csv
import glob
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Plot defaults from style_guide/plots.md and style_guide/guardrails.md §T5
_DPI = 200
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
})

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
_FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures',
                        'phase0', 'face_prior')

_DENORM_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
_DENORM_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def _denormalize(x: torch.Tensor) -> np.ndarray:
    """[1, 3, H, W] normalized -> [H, W, 3] in [0, 1]."""
    arr = x.squeeze(0).cpu().numpy()
    arr = arr * _DENORM_STD + _DENORM_MEAN
    return np.clip(arr, 0, 1).transpose(1, 2, 0)


def _arm_from_tag(tag: str) -> Optional[str]:
    m = re.search(r'face_prior_([A-Z][0-9]?)', tag or '')
    return m.group(1) if m else None


def _load_arm(path: str) -> Dict:
    """Load one sweep .pth, return a dict with arm/config/metrics/tensors."""
    d = torch.load(path, map_location='cpu', weights_only=False)
    tag = d.get('snapshot_dir', '') or path
    arm = _arm_from_tag(tag) or _arm_from_tag(os.path.basename(path))
    return {
        'arm': arm or '?',
        'path': path,
        'tv_weight': d.get('tv_weight'),
        'cos_weight': d.get('cos_weight', 1.0),
        'face_weight': d.get('face_weight', 0.0),
        'metrics': d.get('metrics', {}),
        'x_true': d['x_true'],
        'x_recon': d['x_recon'],
        'loss_history': d.get('loss_history', {}),
        'snapshot_dir': d.get('snapshot_dir'),
    }


def discover_arms(results_dir: str = _RESULTS_DIR) -> List[Dict]:
    """Find all face-prior sweep .pth files and return their parsed contents."""
    pattern = os.path.join(results_dir, 'phase0_full_*face_prior_*.pth')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No face-prior sweep results found at {pattern}. "
            f"Run scripts/run_phase0_face_prior_sweep.sh first."
        )
    return [_load_arm(p) for p in paths]


# ------------------ Figures ------------------

def _draw_landmarks(ax, x_recon_t: torch.Tensor) -> Optional[float]:
    """Run kornia detector on x_recon, scatter top-1 5 keypoints. Returns conf."""
    try:
        from experiments.face_prior import load_face_prior
        prior = load_face_prior(model='auto', device='cpu')
    except Exception as e:
        ax.text(0.02, 0.98, f'(no prior: {type(e).__name__})',
                transform=ax.transAxes, color='red', fontsize=8,
                verticalalignment='top')
        return None
    detector = prior['detector']
    x_pixel = (x_recon_t * torch.tensor(_DENORM_STD).reshape(1, 3, 1, 1)
               + torch.tensor(_DENORM_MEAN).reshape(1, 3, 1, 1)).clamp(0, 1)
    with torch.no_grad():
        dets = detector(x_pixel * 255.0)
    if dets.numel() == 0:
        ax.text(0.02, 0.98, 'no face detected', transform=ax.transAxes,
                color='red', fontsize=9, verticalalignment='top')
        return 0.0
    top = dets[dets[:, 14].argmax()]
    bx1, by1, bx2, by2 = top[:4].tolist()
    ax.add_patch(plt.Rectangle((bx1, by1), bx2 - bx1, by2 - by1,
                                fill=False, edgecolor='lime', linewidth=1.5))
    coords = top[4:14].reshape(5, 2).tolist()
    colors = ['cyan', 'cyan', 'yellow', 'magenta', 'magenta']
    labels = ['eye_l', 'eye_r', 'nose', 'mouth_l', 'mouth_r']
    for (px, py), c, lbl in zip(coords, colors, labels):
        ax.scatter(px, py, c=c, s=30, edgecolor='black', linewidth=0.5,
                   label=lbl)
    conf = float(top[14].item())
    ax.text(0.02, 0.98, f'conf={conf:.2f}', transform=ax.transAxes,
            color='lime', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.4))
    return conf


def make_grid(arms: List[Dict], save_path: str) -> None:
    """Qualitative figure: rows=arms, cols=[GT, recon, recon+landmarks]."""
    n = len(arms)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), dpi=_DPI)
    if n == 1:
        axes = axes[None, :]
    fig.suptitle('Face-prior ablation: per-arm reconstructions on face1.jpg',
                  y=1.005)
    for r, arm in enumerate(arms):
        gt = _denormalize(arm['x_true'])
        rc = _denormalize(arm['x_recon'])
        ssim = arm['metrics'].get('ssim', 0)
        cos = arm['metrics'].get('best_cos_sim', 0)
        face_det = arm['metrics'].get('face_det_score')
        row_label = (f"{arm['arm']}\ntv={arm['tv_weight']}\n"
                     f"cos={arm['cos_weight']}\nface={arm['face_weight']}")

        axes[r, 0].imshow(gt)
        axes[r, 0].set_ylabel(row_label, fontsize=10, rotation=0,
                              labelpad=50, ha='right', va='center')
        if r == 0:
            axes[r, 0].set_title('Ground truth')
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])

        axes[r, 1].imshow(rc)
        ttl = (f"SSIM={ssim:.3f}, cos={cos:.3f}"
               + (f", face_det={face_det:.2f}"
                  if face_det is not None else ''))
        if r == 0:
            axes[r, 1].set_title('Reconstruction')
        axes[r, 1].set_xlabel(ttl, fontsize=9)
        axes[r, 1].set_xticks([])
        axes[r, 1].set_yticks([])

        axes[r, 2].imshow(rc)
        _draw_landmarks(axes[r, 2], arm['x_recon'])
        if r == 0:
            axes[r, 2].set_title('Recon + landmarks')
        axes[r, 2].set_xticks([])
        axes[r, 2].set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')
    plt.close()
    print(f'  saved {save_path}')


def _line_panel(ax, xs: List[float], ys: List[float], ylabel: str,
                color: str, log_x: bool = True) -> None:
    ax.plot(xs, ys, marker='o', linewidth=2, color=color)
    for x, y in zip(xs, ys):
        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                    xytext=(0, 7), fontsize=9, ha='center')
    if log_x:
        ax.set_xscale('log')
    ax.set_ylabel(ylabel, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.spines['top'].set_visible(False)
    ax.grid(True, alpha=0.3)


def _make_strength_or_cos_plot(arms: List[Dict], xkey: str, xlabel: str,
                               save_path: str, log_x: bool = True) -> None:
    """Twin-axis line plot: SSIM (left) and face_det_score (right) vs xkey."""
    rows = sorted(arms, key=lambda a: a[xkey])
    xs = [a[xkey] for a in rows]
    ssims = [a['metrics'].get('ssim', 0) for a in rows]
    fds = [a['metrics'].get('face_det_score', 0) for a in rows]
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=_DPI)
    _line_panel(ax1, xs, ssims, 'SSIM', '#1f77b4', log_x=log_x)
    ax2 = ax1.twinx()
    _line_panel(ax2, xs, fds, 'Face detection score', '#d62728', log_x=log_x)
    ax2.spines['top'].set_visible(False)
    ax1.set_xlabel(xlabel)
    ax1.set_title(f'Phase 0 face prior: SSIM vs face_det_score across {xlabel}')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')
    plt.close()
    print(f'  saved {save_path}')


def make_strength_sweep(arms: List[Dict], save_path: str) -> None:
    """E1, C, D, E4 — face_weight sweep at fixed tv=1e-1, cos=1.0."""
    rows = [a for a in arms if a['tv_weight'] == 0.1
            and abs(a['cos_weight'] - 1.0) < 1e-6
            and a['face_weight'] > 0]
    if len(rows) < 2:
        print('  (skipping strength sweep — need >=2 face_weight points)')
        return
    _make_strength_or_cos_plot(rows, 'face_weight', 'face_weight',
                                save_path, log_x=True)


def make_cos_sweep(arms: List[Dict], save_path: str) -> None:
    """F1, D, F3, F4 — cos_weight sweep at fixed tv=1e-1, face=1e-1."""
    rows = [a for a in arms if a['tv_weight'] == 0.1
            and abs(a['face_weight'] - 0.1) < 1e-6]
    if len(rows) < 2:
        print('  (skipping cos sweep — need >=2 cos_weight points)')
        return
    _make_strength_or_cos_plot(rows, 'cos_weight', 'cos_weight',
                                save_path, log_x=False)


def make_loss_curves(arms: List[Dict], save_path: str) -> None:
    """Stacked loss curves per arm. Best-restart only, log-y for sub-losses."""
    panels = ['cos_sim', 'tv', 'face_total', 'face_layout']
    n_arms = len(arms)
    fig, axes = plt.subplots(n_arms, len(panels),
                              figsize=(4 * len(panels), 3 * n_arms), dpi=_DPI)
    if n_arms == 1:
        axes = axes[None, :]
    for r, arm in enumerate(arms):
        lh = arm['loss_history']
        for c, key in enumerate(panels):
            ax = axes[r, c]
            if key not in lh or not lh[key]:
                ax.text(0.5, 0.5, '—', ha='center', va='center',
                        transform=ax.transAxes, fontsize=20, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(key)
                continue
            best_r = max(range(len(lh['cos_sim'])),
                         key=lambda i: max(lh['cos_sim'][i]))
            vals = lh[key][best_r]
            ax.plot(vals, color='C0' if c == 0 else 'C1', linewidth=1.0)
            if r == 0:
                ax.set_title(key)
            if c == 0:
                ax.set_ylabel(arm['arm'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3)
    fig.suptitle('Face-prior loss curves per arm (best restart)', y=1.001)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')
    plt.close()
    print(f'  saved {save_path}')


def make_landmark_evolution(winner: Dict, save_path: str) -> None:
    """4 snapshots of the winner with landmarks overlaid."""
    snap_dir = winner.get('snapshot_dir')
    if not snap_dir or not os.path.isdir(snap_dir):
        print(f'  (skipping landmark evolution — snapshot dir missing: {snap_dir})')
        return
    files = sorted(glob.glob(os.path.join(snap_dir, 'restart0_iter*.png')))
    if len(files) < 4:
        print(f'  (skipping landmark evolution — fewer than 4 snapshots in {snap_dir})')
        return
    pick = [files[0], files[len(files) // 3], files[2 * len(files) // 3], files[-1]]
    from PIL import Image
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=_DPI)
    for ax, p in zip(axes, pick):
        img = np.array(Image.open(p)) / 255.0
        ax.imshow(img)
        # Re-derive a tensor for landmark detection
        t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        x_norm = ((t - torch.tensor(_DENORM_MEAN).reshape(1, 3, 1, 1))
                  / torch.tensor(_DENORM_STD).reshape(1, 3, 1, 1))
        _draw_landmarks(ax, x_norm)
        m = re.search(r'iter(\d+)', os.path.basename(p))
        ax.set_title(f"iter {int(m.group(1)) if m else '?'}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Winner ({winner['arm']}): landmark evolution", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')
    plt.close()
    print(f'  saved {save_path}')


def write_metrics_csv(arms: List[Dict], save_path: str) -> None:
    fields = ['arm', 'tv_weight', 'cos_weight', 'face_weight',
              'ssim', 'psnr', 'mse', 'best_cos_sim', 'face_det_score']
    with open(save_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for a in arms:
            row = {
                'arm': a['arm'],
                'tv_weight': a['tv_weight'],
                'cos_weight': a['cos_weight'],
                'face_weight': a['face_weight'],
                **{k: a['metrics'].get(k, '') for k in
                   ('ssim', 'psnr', 'mse', 'best_cos_sim', 'face_det_score')},
            }
            w.writerow(row)
    print(f'  saved {save_path}')


def main():
    arms = discover_arms()
    arms = sorted(arms, key=lambda a: (a['arm']))
    print(f'Loaded {len(arms)} arms: {[a["arm"] for a in arms]}')

    os.makedirs(_FIG_DIR, exist_ok=True)
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    make_grid(arms, os.path.join(_FIG_DIR, 'face_prior_grid.png'))
    make_strength_sweep(
        arms, os.path.join(_FIG_DIR, 'face_prior_strength_sweep.png'))
    make_cos_sweep(arms, os.path.join(_FIG_DIR, 'face_prior_cos_sweep.png'))
    make_loss_curves(arms, os.path.join(_FIG_DIR, 'face_prior_loss_curves.png'))

    winner = max(arms, key=lambda a: a['metrics'].get('ssim', -1))
    print(f"\nWinner by SSIM: arm {winner['arm']} "
          f"(SSIM={winner['metrics'].get('ssim', 0):.4f}, "
          f"face_det={winner['metrics'].get('face_det_score', 0):.3f})")
    make_landmark_evolution(
        winner, os.path.join(_FIG_DIR, 'face_prior_landmark_evolution.png'))

    write_metrics_csv(
        arms, os.path.join(_RESULTS_DIR, f'phase0_face_prior_sweep_{ts}.csv'))


if __name__ == '__main__':
    main()
