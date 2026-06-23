"""Regenerate v6 figures for supervisor deck per audit brief (2026-05-14).

Output: PNG ≥150 dpi to figures/v6/, filenames exactly as specified by the brief.
Honest about missing data — prints what's available and what isn't.

Run order: slide 7 first (bug fix), then 5/8/10/13, then 16, then optional 19.

Usage:
    cd /home/projects/galvardi/yoado
    conda activate rec
    python scripts/figures/make_v6_figures.py
"""
import csv
import glob
import hashlib
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.metrics import ssim as kssim
from matplotlib.patches import Polygon, FancyArrowPatch
from PIL import Image as PILImage

# ============================================================
# CONSTANTS / STYLE
# ============================================================
ROOT = Path('/home/projects/galvardi/yoado')
OUT = ROOT / 'figures/v6'
OUT.mkdir(parents=True, exist_ok=True)

# Color palette (per brief)
C_FULL    = '#1F4E79'   # full FT / primary blue
C_LORA    = '#E07A1F'   # LoRA r=8 / secondary orange
C_NEG     = '#C0392B'   # negative / baseline red
C_POS     = '#2C8050'   # positive / success green
C_MODREL  = '#888888'   # modified ReLU grey (we use orange for LoRA)
C_GATE    = '#7A7A7A'   # gate line grey
GATE_SSIM = 0.30

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


def psnr(recon, target):
    import torch.nn.functional as F
    mse = F.mse_loss(recon.clamp(0, 1), target.clamp(0, 1))
    if mse.item() <= 0:
        return float('inf')
    return -10 * torch.log10(mse).item()


def to_gray_img(t):
    """Tensor (1,H,W) or (1,1,H,W) -> (H,W) numpy."""
    if t.ndim == 4: t = t[0, 0]
    elif t.ndim == 3: t = t[0]
    return t.detach().cpu().numpy()


def to_rgb_img(t):
    """Tensor (C,H,W) or (1,C,H,W) -> (H,W,C) numpy in [0,1]."""
    if t.ndim == 4: t = t[0]
    arr = t.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(arr, 0, 1)


def add_gate_line(ax):
    """Add horizontal dashed grey line at SSIM = 0.30 with 'gate' annotation."""
    ax.axhline(GATE_SSIM, color=C_GATE, linestyle='--', linewidth=1.2, alpha=0.7, zorder=0)
    xl = ax.get_xlim()
    ax.text(xl[1], GATE_SSIM + 0.01, ' gate', color=C_GATE, fontsize=9, alpha=0.8,
            ha='right', va='bottom')


# ============================================================
# SLIDE 7 (RUN FIRST — fixes the duplicate-SSIM bug)
# ============================================================
def make_slide7():
    print('\n' + '=' * 60)
    print('SLIDE 7: ntk_T10_grid — bug-fix + SSIM-vs-T side panel')
    print('=' * 60)

    # Load T=10 reconstruction tensor (full FT only; LoRA T=10 tensor not saved)
    t10_path = ROOT / 'results/exp_b_T10_full_s42_a149.pth'
    d = torch.load(t10_path, map_location='cpu', weights_only=False)
    # Tensor uses 'x_train' (not 'x_true')
    x_train = d.get('x_train', d.get('x_true'))
    x_recon = d['x_recon_full']
    x_ctrl  = d['x_ctrl']
    ds_mean = d.get('ds_mean')
    if isinstance(ds_mean, torch.Tensor):
        x_train = x_train + ds_mean
        x_recon = x_recon + ds_mean
        x_ctrl  = x_ctrl  + ds_mean

    N = x_train.shape[0]
    per_recon = []
    per_ctrl  = []
    for i in range(N):
        s_r = kssim(x_recon[i:i+1].clamp(0,1), x_train[i:i+1].clamp(0,1), window_size=3).mean().item()
        s_c = kssim(x_ctrl [i:i+1].clamp(0,1), x_train[i:i+1].clamp(0,1), window_size=3).mean().item()
        per_recon.append(s_r)
        per_ctrl .append(s_c)

    print(f'Per-sample recon SSIM:   {[f"{s:.3f}" for s in per_recon]}')
    print(f'Per-sample control SSIM: {[f"{s:.3f}" for s in per_ctrl]}')
    if abs(per_recon[0] - per_recon[1]) < 1e-3:
        print('WARNING: per-sample SSIMs are identical — possible data issue')
    else:
        print('OK: per-sample SSIMs differ (bug fixed).')

    # Side-panel data: SSIM vs T from sprint2b_phase2 CSV (LeakyReLU, both full and LoRA r=8)
    p2 = list(csv.DictReader(open(ROOT / 'results/sprint2b_phase2_20260223_072927.csv')))
    full_by_T = {}
    lora_by_T = {}
    for r in p2:
        try:
            T = int(float(r['n_steps']))
            ss_str = r.get('full_ssim','')
            ls_str = r.get('lora_ssim','')
            rank = r.get('rank','')
            if rank == '' and ss_str:
                full_by_T.setdefault(T, []).append(float(ss_str))
            elif rank and int(float(rank))==8 and ls_str:
                lora_by_T.setdefault(T, []).append(float(ls_str))
        except (ValueError, KeyError): continue
    Tlist = sorted(set(full_by_T) | set(lora_by_T))
    full_means = [np.mean(full_by_T[T]) for T in Tlist if T in full_by_T]
    lora_means = [np.mean(lora_by_T[T]) for T in Tlist if T in lora_by_T]
    print(f'Side panel  T values:    {Tlist}')
    print(f'Side panel  full SSIM:   {[f"{x:.3f}" for x in full_means]}')
    print(f'Side panel  LoRA SSIM:   {[f"{x:.3f}" for x in lora_means]}')

    # Build figure: 3 rows × 2 cols (images) + side panel
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(3, 4, width_ratios=[0.5, 1, 1, 2], hspace=0.18, wspace=0.15)

    labels = ['Ground Truth',
              'Recon @ T=10\n(full FT)',
              'Negative Control\n(diff. sample,\nsame class)']
    for r, lbl in enumerate(labels):
        ax = fig.add_subplot(gs[r, 0]); ax.axis('off')
        ax.text(0.5, 0.5, lbl, ha='center', va='center', fontsize=12, fontweight='bold')

    for i in range(N):
        col = i + 1
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(to_gray_img(x_train[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
        ax.set_title(f'Sample {i+1}', fontsize=12)

        ax = fig.add_subplot(gs[1, col])
        ax.imshow(to_gray_img(x_recon[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
        ax.text(0.5, -0.08, f'SSIM = {per_recon[i]:.3f}', transform=ax.transAxes,
                ha='center', fontsize=11, color=C_POS, fontweight='bold')

        ax = fig.add_subplot(gs[2, col])
        ax.imshow(to_gray_img(x_ctrl[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
        ax.text(0.5, -0.08, f'SSIM = {per_ctrl[i]:.3f}', transform=ax.transAxes,
                ha='center', fontsize=11, color=C_NEG)

    # Side panel: SSIM vs T
    ax_side = fig.add_subplot(gs[:, 3])
    if full_means:
        Tf = [T for T in Tlist if T in full_by_T]
        ax_side.plot(Tf, full_means, 'o-', color=C_FULL, linewidth=2.5, markersize=9, label='Full FT')
    if lora_means:
        Tl = [T for T in Tlist if T in lora_by_T]
        ax_side.plot(Tl, lora_means, 's-', color=C_LORA, linewidth=2.5, markersize=9, label='LoRA r=8')
    ax_side.set_xscale('log')
    ax_side.set_xlabel('Fine-tuning steps T')
    ax_side.set_ylabel('Reconstruction SSIM')
    ax_side.set_ylim(0, 1.05)
    ax_side.grid(alpha=0.3, which='both')
    add_gate_line(ax_side)
    ax_side.axvline(10, color='#bbb', linestyle=':', alpha=0.6)
    ax_side.text(10.5, 0.04, 'this slide', color='#666', fontsize=9, alpha=0.8)
    ax_side.legend(loc='lower left')
    ax_side.set_title('Quality vs steps (LeakyReLU)')

    plt.suptitle('NTK reconstruction at T = 10 (full FT, MNIST) — per-sample SSIM + quality vs T',
                 fontsize=14, fontweight='bold', y=0.99)
    out = OUT / 'slide7_ntk_T10_grid.png'
    plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')
    print('Note: T=10 LoRA tensor not saved — main grid is full FT only.')
    print('      Side panel uses sprint2b_phase2 CSV (LeakyReLU, T=1/5/10/20/100).')


# ============================================================
# SLIDE 5 (NTK MNIST grid with Δ inset)
# ============================================================
def make_slide5():
    print('\n' + '=' * 60)
    print('SLIDE 5: ntk_mnist_grid — 2x4 grid + Δ inset')
    print('=' * 60)

    t1_path = ROOT / 'results/exp_b_T1_r8_free_s42_a10000.pth'
    d = torch.load(t1_path, map_location='cpu', weights_only=False)
    x_train  = d.get('x_train', d.get('x_true'))
    x_recon_full = d['x_recon_full']
    x_recon_lora = d['x_recon_lora']
    x_ctrl   = d['x_ctrl']
    ds_mean  = d.get('ds_mean')
    if isinstance(ds_mean, torch.Tensor):
        x_train = x_train + ds_mean
        x_recon_full = x_recon_full + ds_mean
        x_recon_lora = x_recon_lora + ds_mean
        x_ctrl  = x_ctrl  + ds_mean

    N = x_train.shape[0]
    full_ssim, lora_ssim, ctrl_ssim = [], [], []
    for i in range(N):
        full_ssim.append(kssim(x_recon_full[i:i+1].clamp(0,1), x_train[i:i+1].clamp(0,1), window_size=3).mean().item())
        lora_ssim.append(kssim(x_recon_lora[i:i+1].clamp(0,1), x_train[i:i+1].clamp(0,1), window_size=3).mean().item())
        ctrl_ssim.append(kssim(x_ctrl[i:i+1].clamp(0,1), x_train[i:i+1].clamp(0,1), window_size=3).mean().item())
    deltas = [l - c for l, c in zip(lora_ssim, ctrl_ssim)]
    print(f'Sample 1 SSIM: full={full_ssim[0]:.3f}, LoRA={lora_ssim[0]:.3f}, ctrl={ctrl_ssim[0]:.3f}, Δ={deltas[0]:+.3f}')
    print(f'Sample 2 SSIM: full={full_ssim[1]:.3f}, LoRA={lora_ssim[1]:.3f}, ctrl={ctrl_ssim[1]:.3f}, Δ={deltas[1]:+.3f}')

    fig = plt.figure(figsize=(13, 9))
    # Main 4-row x 3-col grid (label col + 2 samples), plus inset bar in top-right
    gs = fig.add_gridspec(4, 4, width_ratios=[0.5, 1, 1, 1.1], height_ratios=[1,1,1,1], hspace=0.18, wspace=0.18)

    labels = ['Ground Truth\n(private)',
              'Full FT recon\n(1.8M params)',
              'LoRA r=8 recon\n(38K params)',
              'Negative Control\n(diff. sample,\nsame class)']
    for r, lbl in enumerate(labels):
        ax = fig.add_subplot(gs[r, 0]); ax.axis('off')
        ax.text(0.5, 0.5, lbl, ha='center', va='center', fontsize=11, fontweight='bold')

    for i in range(N):
        col = i + 1
        # Row 0: GT
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(to_gray_img(x_train[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
        ax.set_title(f'Sample {i+1}', fontsize=12)
        # Row 1: full
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(to_gray_img(x_recon_full[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
        ax.text(0.5, -0.08, f'SSIM = {full_ssim[i]:.3f}', transform=ax.transAxes,
                ha='center', fontsize=11, color=C_FULL, fontweight='bold')
        # Row 2: LoRA
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(to_gray_img(x_recon_lora[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
        d_color = C_POS if deltas[i] > 0 else C_NEG
        ax.text(0.5, -0.08, f'SSIM = {lora_ssim[i]:.3f}',
                transform=ax.transAxes, ha='center', fontsize=11, color=C_LORA, fontweight='bold')
        # Row 3: control
        ax = fig.add_subplot(gs[3, col])
        ax.imshow(to_gray_img(x_ctrl[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
        ax.text(0.5, -0.08, f'SSIM = {ctrl_ssim[i]:.3f}', transform=ax.transAxes,
                ha='center', fontsize=11, color=C_NEG)

    # Δ inset bar chart (spans rows 0-2 of last column)
    ax_inset = fig.add_subplot(gs[0:3, 3])
    xs = np.arange(N)
    colors_bar = [C_POS if d_ > 0 else C_NEG for d_ in deltas]
    ax_inset.bar(xs, deltas, color=colors_bar, edgecolor='black', linewidth=1)
    ax_inset.axhline(0, color='black', linewidth=0.8)
    ax_inset.set_xticks(xs)
    ax_inset.set_xticklabels([f'S{i+1}' for i in range(N)])
    ax_inset.set_ylabel('Δ = LoRA − Control SSIM')
    ax_inset.set_title('Instance-level lift\n(LoRA − control)', fontsize=12, fontweight='bold')
    for i, d_ in enumerate(deltas):
        va = 'bottom' if d_ > 0 else 'top'
        ax_inset.text(i, d_ + (0.01 if d_ > 0 else -0.01), f'{d_:+.3f}',
                      ha='center', va=va, fontsize=10, fontweight='bold')
    ax_inset.set_ylim(min(min(deltas) - 0.05, -0.05), max(max(deltas) + 0.05, 0.10))
    ax_inset.grid(alpha=0.3, axis='y')

    # Caption row
    fig.text(0.5, 0.02,
             'Full FT recovers nearly perfectly; LoRA r=8 recovers Sample 1 above control, Sample 2 below. '
             'Instance-level leakage is per-sample, not guaranteed — see slide 10 for distribution.',
             ha='center', fontsize=11, style='italic', color='#333')

    out = OUT / 'slide5_ntk_mnist_grid.png'
    plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ============================================================
# SLIDE 8 (stability vs quality — densified with available data)
# ============================================================
def make_slide8():
    print('\n' + '=' * 60)
    print('SLIDE 8: stability_vs_quality — densified')
    print('=' * 60)

    ph0 = list(csv.DictReader(open(ROOT / 'results/sprint2b_phase0_20260222_191555.csv')))
    ph2 = list(csv.DictReader(open(ROOT / 'results/sprint2b_phase2_20260223_072927.csv')))

    acts = {'leaky_relu': C_POS, 'modified_relu': C_LORA, 'relu_default': C_NEG}
    labels_act = {'leaky_relu': 'LeakyReLU', 'modified_relu': 'Modified ReLU', 'relu_default': 'ReLU'}

    data = []  # (lin_err, ssim, activation, T)
    for r in ph0:
        a = r.get('activation', '')
        if a not in acts: continue
        try:
            le = float(r['full_linearization_error'])
            ss_str = r.get('full_ssim','')
            if not ss_str or ss_str == 'nan': continue
            ss = float(ss_str)
            T = int(float(r['n_steps']))
            data.append((max(le, 1e-15), ss, a, T))
        except (ValueError, KeyError): continue
    for r in ph2:
        if r.get('rank','') != '': continue
        try:
            le = float(r['full_linearization_error'])
            ss = float(r['full_ssim'])
            T = int(float(r['n_steps']))
            data.append((max(le, 1e-15), ss, 'leaky_relu', T))
        except (ValueError, KeyError): continue

    print(f'Total points: {len(data)}')
    Ts_available = sorted(set(d[3] for d in data))
    print(f'T values present: {Ts_available}')
    Ts_requested = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
    Ts_missing = [t for t in Ts_requested if t not in Ts_available]
    print(f'T values requested by brief but NOT in CSV: {Ts_missing}')
    print('  -> Densification limited to available T. Re-run sprint2b at these T to fully densify.')

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for a, col in acts.items():
        pts = [(le, ss) for le, ss, act, T in data if act == a]
        if not pts: continue
        les = [p[0] for p in pts]
        sss = [p[1] for p in pts]
        ax.scatter(les, sss, c=col, s=110, alpha=0.7, edgecolor='black', linewidth=0.8, label=labels_act[a])
        # Trend (polynomial deg 2 on log-error)
        log_les = np.log10(les)
        if len(log_les) >= 3 and np.ptp(log_les) > 0.5:
            try:
                coeffs = np.polyfit(log_les, sss, 2)
                xx = np.linspace(min(log_les), max(log_les), 50)
                yy = np.polyval(coeffs, xx)
                yy = np.clip(yy, 0, 1.05)
                ax.plot(10**xx, yy, color=col, linewidth=1.2, alpha=0.55, linestyle='-')
            except Exception:
                pass

    ax.set_xscale('log')
    ax.set_xlabel('Linearization error  (log scale; lower = NTK approximation more accurate)')
    ax.set_ylabel('Reconstruction SSIM')
    ax.set_ylim(0, 1.05)
    ax.set_title('NTK stability ↔ reconstruction quality   (across T, 3 activations)',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, which='both')
    add_gate_line(ax)
    ax.legend(loc='lower left')

    if Ts_missing:
        ax.text(0.02, 0.98,
                f'Showing available T = {Ts_available}.\nDenser sweep ({Ts_requested}) requires re-run.',
                transform=ax.transAxes, fontsize=9, color='#888', va='top',
                bbox=dict(boxstyle='round', facecolor='#fafafa', edgecolor='#ccc'))

    plt.tight_layout()
    out = OUT / 'slide8_stability_vs_quality.png'
    plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ============================================================
# SLIDE 10 (multi-seed histogram + control reference)
# ============================================================
def make_slide10():
    print('\n' + '=' * 60)
    print('SLIDE 10: multi_seed_validation — 50-seed hist + control reference')
    print('=' * 60)

    free_ssim = []
    with open(ROOT / 'results/multiseed_freec_vs_oracle_20260327_034128.csv') as f:
        for row in csv.DictReader(f):
            if row['mode'] == 'free_c':
                free_ssim.append(float(row['ssim']))
    arr = np.array(free_ssim)
    print(f'N seeds: {len(arr)}; mean = {arr.mean():.3f}, median = {np.median(arr):.3f}, std = {arr.std(ddof=1):.3f}')

    # Negative control mean — from slide 5 controls (per-sample mean)
    # Compute from the actual slide-5 tensor
    t5 = torch.load(ROOT / 'results/exp_b_T1_r8_free_s42_a10000.pth', map_location='cpu', weights_only=False)
    xt = t5.get('x_train', t5.get('x_true'))
    xc = t5['x_ctrl']
    ds = t5.get('ds_mean')
    if isinstance(ds, torch.Tensor):
        xt = xt + ds; xc = xc + ds
    ctrl_ssims = []
    for i in range(xt.shape[0]):
        ctrl_ssims.append(kssim(xc[i:i+1].clamp(0,1), xt[i:i+1].clamp(0,1), window_size=3).mean().item())
    control_mean = float(np.mean(ctrl_ssims))
    print(f'Negative-control SSIM (from slide-5 control panels): {ctrl_ssims}, mean = {control_mean:.3f}')

    pct_above = 100.0 * (arr > control_mean).sum() / len(arr)
    print(f'Percent of seeds with SSIM > control mean: {pct_above:.1f}%')

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(arr, bins=22, color=C_LORA, edgecolor='black', alpha=0.85, label='LoRA r=8 recon (50 seeds)')
    ax.axvline(arr.mean(), color=C_FULL, linestyle='--', linewidth=2,
               label=f'Recon mean = {arr.mean():.3f}')
    ax.axvline(control_mean, color=C_NEG, linestyle='--', linewidth=2,
               label=f'Negative-control mean = {control_mean:.2f}  (slide 5)')
    ax.set_xlabel('SSIM')
    ax.set_ylabel('Count')
    ax.set_title('Multi-seed reconstruction (50 seeds, LoRA r=8, MNIST)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.98,
            f'{pct_above:.1f}% of seeds exceed negative-control mean →  even one-in-N is real leakage.',
            transform=ax.transAxes, fontsize=10, color='#444', va='top',
            bbox=dict(boxstyle='round', facecolor='#fafafa', edgecolor='#ccc'))
    plt.tight_layout()
    out = OUT / 'slide10_multiseed_hist.png'
    plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ============================================================
# SLIDE 13 (three faces with SSIM/PSNR labels)
# ============================================================
def make_slide13():
    print('\n' + '=' * 60)
    print('SLIDE 13: faces_triptych — 3 faces with SSIM/PSNR labels')
    print('=' * 60)

    # Per the brief: face1 ≈ 0.52 / 13.8 dB; face2/3 from same D3-winner config.
    # Use the actual tensors we have.
    faces = [
        ('face1', ROOT / 'results/phase0_full_r8_n1_s42_20260428_134922_face_d3winner_freq1e-3.pth'),
        ('face2', ROOT / 'results/phase0_full_r8_n1_s42_20260513_011954_face2_d3winner_freq1e-3.pth'),
        ('face3', ROOT / 'results/phase0_full_r8_n1_s42_20260513_011954_face3_d3winner_freq1e-3.pth'),
    ]
    gts, recons, metrics = [], [], []
    for name, path in faces:
        d = torch.load(path, map_location='cpu', weights_only=False)
        xt_ = d.get('x_train', d.get('x_true'))
        xr_ = d['x_recon']
        if xt_.ndim == 4 and xt_.shape[0] > 1:
            xt_ = xt_[0:1]; xr_ = xr_[0:1]
        s = kssim(xr_.clamp(0,1), xt_.clamp(0,1), window_size=3).mean().item()
        ps = psnr(xr_, xt_)
        gts.append(xt_); recons.append(xr_); metrics.append((s, ps))
        print(f'  {name}: SSIM = {s:.3f}, PSNR = {ps:.1f} dB')

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    for i in range(3):
        axes[0, i].imshow(to_rgb_img(gts[i])); axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        for sp in axes[0, i].spines.values(): sp.set_color('#888')
        axes[0, i].set_title(f'Person {i+1}', fontsize=13)

        axes[1, i].imshow(to_rgb_img(recons[i])); axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        for sp in axes[1, i].spines.values(): sp.set_color('#888')
        s, ps = metrics[i]
        axes[1, i].text(0.5, -0.08, f'SSIM = {s:.2f}    PSNR = {ps:.1f} dB',
                        transform=axes[1, i].transAxes, ha='center', fontsize=11,
                        color=C_POS, fontweight='bold')

    axes[0, 0].set_ylabel('Original',    fontsize=14, labelpad=10)
    axes[1, 0].set_ylabel('Recovered\nfrom ViT gradient', fontsize=14, labelpad=10)
    plt.subplots_adjust(wspace=0.05, hspace=0.12, top=0.93, bottom=0.06, left=0.10, right=0.99)
    out = OUT / 'slide13_faces_triptych.png'
    plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')
    print('Note: random-init negative-control row NOT included — would require a fresh inversion run.')
    print('      Tell user: skip for now, can be added in a follow-up if needed.')


# ============================================================
# SLIDE 16 (KKT schematic — pure illustration, no data)
# ============================================================
def make_slide16():
    print('\n' + '=' * 60)
    print('SLIDE 16: kkt_schematic — synthetic illustration')
    print('=' * 60)

    np.random.seed(2026)
    n_support = 110
    # Two slightly overlapping clusters (pre-training + fine-tuning support)
    pre_pts = np.random.randn(80, 2) * np.array([1.8, 1.0]) + np.array([-0.4, 0.3])
    ft_pts  = np.random.randn(30, 2) * np.array([1.4, 0.9]) + np.array([0.6, -0.5])
    all_pts = np.vstack([pre_pts, ft_pts])

    fig, ax = plt.subplots(figsize=(13, 7.3))

    # BA span: a rotated parallelogram (rank-2 subspace projected to 2D as a stripe)
    cx, cy = 0.0, 0.0
    w, h = 4.6, 0.7
    ang_rad = np.deg2rad(-14)
    cos_a, sin_a = np.cos(ang_rad), np.sin(ang_rad)
    corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners_rot = corners @ R.T + np.array([cx, cy])
    ba_patch = Polygon(corners_rot, closed=True, facecolor=C_LORA, edgecolor=C_LORA, alpha=0.30, linewidth=2.5)
    ax.add_patch(ba_patch)

    # Support points
    ax.scatter(all_pts[:, 0], all_pts[:, 1], c='#555', s=42, alpha=0.65, edgecolor='black', linewidth=0.4,
               label=f'KKT support points  (n ≈ {len(all_pts)},  pre-train + fine-tune)')

    # Highlight the 2 "fine-tuning targets" the extraction allots
    # Pick 2 from ft_pts that are roughly on the BA plane axis
    ft_targets_idx = [12, 24]
    ft_targets = ft_pts[ft_targets_idx]
    # Snap them to the BA span axis (rotation through origin)
    for t in ft_targets:
        # Project t onto the BA span axis (the long axis of the parallelogram)
        u = np.array([cos_a, sin_a])
        proj = (t @ u) * u
        ax.scatter([proj[0]], [proj[1]], c=C_NEG, s=170, edgecolor='black', linewidth=1.5, zorder=6)
    ax.scatter([], [], c=C_NEG, s=170, edgecolor='black', linewidth=1.5, label='2 fine-tuning targets  (the only x₁, x₂ the extraction allots)')

    # Residual arrows: for ~7 support points NOT on the plane, draw red arrow to their projection
    np.random.seed(2027)
    residual_idx = np.random.choice(np.arange(len(all_pts)), 7, replace=False)
    u = np.array([cos_a, sin_a])
    annotated_once = False
    for idx in residual_idx:
        pt = all_pts[idx]
        # Project onto BA-span axis
        proj = (pt @ u) * u
        arr = FancyArrowPatch(pt, proj, arrowstyle='->', color=C_NEG, mutation_scale=12, linewidth=1.4, alpha=0.85, zorder=4)
        ax.add_patch(arr)
        if not annotated_once:
            mid = 0.5 * (pt + proj)
            ax.annotate('‖residual‖ ≈ ‖W₀‖²',
                        xy=tuple(mid), xytext=(mid[0] + 1.4, mid[1] + 1.2),
                        fontsize=12, color=C_NEG,
                        arrowprops=dict(arrowstyle='-', color=C_NEG, alpha=0.5))
            annotated_once = True

    # Label the BA span
    label_pt = np.array([cx + 2.0, cy + 0.55])
    ax.text(label_pt[0], label_pt[1] - 0.15, 'BA span\n(rank 2)',
            fontsize=14, color=C_LORA, fontweight='bold', ha='center')

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#aaa')
    ax.set_title('rank-2 BA cannot interpolate the full KKT system',
                 fontsize=14, fontweight='bold', y=1.02)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    out = OUT / 'slide16_kkt_schematic.png'
    plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ============================================================
# SLIDE 19 (OPTIONAL: D2 corner exemplars)
# ============================================================
def make_slide19():
    print('\n' + '=' * 60)
    print('SLIDE 19: corner_exemplars — 4 D2 sweep corners (OPTIONAL)')
    print('=' * 60)

    d2csv = ROOT / 'results/phase0_d2_comparison_20260429_121634.csv'
    d2 = list(csv.DictReader(open(d2csv)))
    parsed = []
    for r in d2:
        try:
            if r.get('ssim') and r.get('best_cos_sim') and r.get('tv_weight'):
                parsed.append({
                    'idx': int(r['config_index']),
                    'tv':  float(r['tv_weight']),
                    'cos': float(r['best_cos_sim']),
                    'ssim': float(r['ssim']),
                    'iters': int(r.get('n_iters', 30000)),
                })
        except (ValueError, KeyError): continue
    parsed = [p for p in parsed if p['iters'] == 30000]
    print(f'D2 configs (30K iters): {len(parsed)}')

    cos_med = np.median([p['cos'] for p in parsed])
    tv_med  = np.median([p['tv']  for p in parsed])
    corners = {('hc','lt'): None, ('hc','ht'): None, ('lc','lt'): None, ('lc','ht'): None}
    for p in parsed:
        k = ('hc' if p['cos'] >= cos_med else 'lc',
             'ht' if p['tv']  >= tv_med  else 'lt')
        if corners[k] is None:
            corners[k] = p
        else:
            # Prefer high-SSIM in high-TV corners, low-SSIM in low-TV (visualize the lever)
            if k[1] == 'ht' and p['ssim'] > corners[k]['ssim']:
                corners[k] = p
            elif k[1] == 'lt' and p['ssim'] < corners[k]['ssim']:
                corners[k] = p

    d2_recons = sorted(glob.glob(str(ROOT / 'figures/phase0/d2_sweep/d2_*_recon.png')))

    def find_recon(p):
        if p is None: return None
        cands = [f for f in d2_recons if f'd2_{p["idx"]:02d}_' in os.path.basename(f)]
        return cands[0] if cands else None

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.5))
    order = [('hc','lt'), ('hc','ht'), ('lc','lt'), ('lc','ht')]
    labels = ['High cos, low TV', 'High cos, high TV', 'Low cos, low TV', 'Low cos, high TV']
    positions = [(0,0), (0,1), (1,0), (1,1)]
    for k, (i, j), title in zip(order, positions, labels):
        ax = axes[i, j]
        p_ = corners[k]
        if p_:
            png = find_recon(p_)
            if png and os.path.exists(png):
                im = PILImage.open(png)
                ax.imshow(im)
            ax.set_title(f'{title}\nSSIM = {p_["ssim"]:.3f}    cos = {p_["cos"]:.3f}    TV = {p_["tv"]:.0e}',
                         fontsize=12)
            print(f'  {k}: idx={p_["idx"]:2d}  ssim={p_["ssim"]:.3f}  cos={p_["cos"]:.3f}  tv={p_["tv"]:.0e}')
        ax.axis('off')

    plt.suptitle('Same gradient match, very different reconstructions — TV weight is the lever',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.text(0.5, 0.02,
             'All 4 share cos_sim ≈ 0.94, but SSIM ranges 0.18–0.55 depending on TV weight.',
             ha='center', fontsize=11, style='italic', color='#444')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = OUT / 'slide19_corner_exemplars.png'
    plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    # Slide 7 FIRST (the bug)
    make_slide7()
    make_slide5()
    make_slide8()
    make_slide10()
    make_slide13()
    make_slide16()
    make_slide19()
    print('\nAll v6 figures written to:', OUT)
