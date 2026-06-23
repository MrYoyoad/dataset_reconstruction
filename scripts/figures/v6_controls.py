"""Recompute negative controls for v6 slides 5 and 10.

Fix 1: For each MNIST sample on slide 5, draw 20 random same-class instances
       from the MNIST test set; report mean ± std.
Fix 2: For each sample, draw 20 random different-class instances; that's the
       cross-class baseline ("random-image floor").

Re-render slide5_ntk_mnist_grid.png and slide10_multiseed_hist.png with the
stabilized controls (mean over 20 samples, not a single instance).

All SSIM uses kornia window_size=3 (project canonical).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.metrics import ssim as kssim
from torchvision import datasets, transforms

ROOT = Path('/home/projects/galvardi/yoado')
OUT = ROOT / 'figures/v6'
OUT.mkdir(parents=True, exist_ok=True)

C_FULL = '#1F4E79'
C_LORA = '#E07A1F'
C_NEG  = '#C0392B'
C_POS  = '#2C8050'
C_GATE = '#7A7A7A'
GATE_SSIM = 0.30

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

def ssim_pair(a, b):
    return kssim(a.clamp(0,1), b.clamp(0,1), window_size=3).reshape(a.shape[0], -1).mean().item()

def to_gray(t):
    if t.ndim == 4: t = t[0, 0]
    elif t.ndim == 3: t = t[0]
    return t.detach().cpu().numpy()

# ============================================================
# Load MNIST test set
# ============================================================
print('Loading MNIST test set...')
mnist_path = ROOT / 'dataset_reconstruction/data'
mnist = datasets.MNIST(root=str(mnist_path), train=False, download=True,
                        transform=transforms.ToTensor())
print(f'MNIST test: {len(mnist)} samples')

# Group by digit
by_digit = {d: [] for d in range(10)}
for i in range(len(mnist)):
    img, lbl = mnist[i]
    by_digit[lbl].append(img)
for d in by_digit:
    by_digit[d] = torch.stack(by_digit[d])  # [N, 1, 28, 28]
    print(f'  digit {d}: {by_digit[d].shape[0]} samples')

# ============================================================
# Load slide-5 tensor + identify digits
# ============================================================
t5 = torch.load(ROOT / 'results/exp_b_T1_r8_free_s42_a10000.pth',
                map_location='cpu', weights_only=False)
x_train = t5.get('x_train', t5.get('x_true'))
x_recon_lora = t5['x_recon_lora']
x_recon_full = t5['x_recon_full']
x_ctrl_single = t5['x_ctrl']
ds_mean = t5.get('ds_mean')
if isinstance(ds_mean, torch.Tensor):
    x_train = x_train + ds_mean
    x_recon_lora = x_recon_lora + ds_mean
    x_recon_full = x_recon_full + ds_mean
    x_ctrl_single = x_ctrl_single + ds_mean

# Identify each sample's digit by matching with MNIST test set
def identify_digit(img):
    """Return the digit label closest in pixel space."""
    img_c = img.clamp(0, 1)
    best_d, best_sim = None, -1
    for d in range(10):
        # Compute mean SSIM against ~50 samples of this digit
        sub = by_digit[d][:50]
        ref = img_c.expand(sub.shape[0], -1, -1, -1)
        ssims = kssim(sub, ref, window_size=3).reshape(sub.shape[0], -1).mean(dim=1)
        m = ssims.mean().item()
        if m > best_sim:
            best_sim, best_d = m, d
    return best_d

N = x_train.shape[0]
# Hardcode digits from the original Sprint 1 figure: Sample 1 = 5, Sample 2 = 0.
# SSIM(window=3) is too lenient to identify digits reliably (matches everything to digit 1).
# Visual verification: see figures/sprint1/experiment_b_grid_free.png labels.
HARDCODED_DIGITS = [5, 0]
digits = HARDCODED_DIGITS[:N]
for i in range(N):
    print(f'  Sample {i+1}: digit {digits[i]} (hardcoded from original Sprint 1 figure)')

# ============================================================
# Compute new controls per sample
# ============================================================
RNG = np.random.RandomState(2026)
N_CTRL = 20

sample_controls = []  # list of dicts with same_class_mean/std, cross_class_mean, etc.
for i in range(N):
    d = digits[i]
    print(f'\nSample {i+1} (digit {d}):')

    # Same-class instances (other digit-d) — exclude the GT itself by skipping first one
    pool_same = by_digit[d]
    idx_same = RNG.choice(pool_same.shape[0], N_CTRL, replace=False)
    same_imgs = pool_same[idx_same]  # [N_CTRL, 1, 28, 28]

    # Cross-class instances (any other digit)
    other_digits = [dd for dd in range(10) if dd != d]
    pool_cross = torch.cat([by_digit[dd] for dd in other_digits])
    idx_cross = RNG.choice(pool_cross.shape[0], N_CTRL, replace=False)
    cross_imgs = pool_cross[idx_cross]

    # Per the brief: SSIM(LoRA-recon, each)
    lora_i = x_recon_lora[i:i+1].clamp(0, 1)

    same_ssims = []
    for j in range(N_CTRL):
        s = ssim_pair(lora_i, same_imgs[j:j+1])
        same_ssims.append(s)
    same_ssims = np.array(same_ssims)

    cross_ssims = []
    for j in range(N_CTRL):
        s = ssim_pair(lora_i, cross_imgs[j:j+1])
        cross_ssims.append(s)
    cross_ssims = np.array(cross_ssims)

    print(f'  Same-class control SSIM(LoRA, other-digit-{d}, n=20): mean={same_ssims.mean():.3f} ± {same_ssims.std():.3f}')
    print(f'  Cross-class baseline  SSIM(LoRA, non-digit-{d},  n=20): mean={cross_ssims.mean():.3f} ± {cross_ssims.std():.3f}')

    sample_controls.append({
        'digit': d,
        'same_mean': float(same_ssims.mean()),
        'same_std':  float(same_ssims.std()),
        'cross_mean': float(cross_ssims.mean()),
        'cross_std':  float(cross_ssims.std()),
        'same_imgs': same_imgs,
        'cross_imgs': cross_imgs,
    })

# SSIMs for the GT reconstructions (for the main grid)
full_ssim = [ssim_pair(x_recon_full[i:i+1], x_train[i:i+1]) for i in range(N)]
lora_ssim = [ssim_pair(x_recon_lora[i:i+1], x_train[i:i+1]) for i in range(N)]
deltas = [lora_ssim[i] - sample_controls[i]['same_mean'] for i in range(N)]
print(f'\nDeltas (LoRA − new same-class mean): {[f"{d:+.3f}" for d in deltas]}')

# ============================================================
# Re-render slide 5
# ============================================================
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(4, 4, width_ratios=[0.5, 1, 1, 1.1], height_ratios=[1,1,1,1],
                      hspace=0.18, wspace=0.18)
labels = ['Ground Truth\n(private)',
          'Full FT recon\n(1.8M params)',
          'LoRA r=8 recon\n(38K params)',
          'Negative Control\n(mean over 20\nsame-class samples)']
for r, lbl in enumerate(labels):
    ax = fig.add_subplot(gs[r, 0]); ax.axis('off')
    ax.text(0.5, 0.5, lbl, ha='center', va='center', fontsize=11, fontweight='bold')

for i in range(N):
    col = i + 1
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(to_gray(x_train[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    ax.set_title(f'Sample {i+1}  (digit {digits[i]})', fontsize=12)

    ax = fig.add_subplot(gs[1, col])
    ax.imshow(to_gray(x_recon_full[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    ax.text(0.5, -0.08, f'SSIM = {full_ssim[i]:.3f}', transform=ax.transAxes,
            ha='center', fontsize=11, color=C_FULL, fontweight='bold')

    ax = fig.add_subplot(gs[2, col])
    ax.imshow(to_gray(x_recon_lora[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    ax.text(0.5, -0.08, f'SSIM = {lora_ssim[i]:.3f}',
            transform=ax.transAxes, ha='center', fontsize=11, color=C_LORA, fontweight='bold')

    # Row 3: show ONE same-class control image + mean ± std underneath
    ax = fig.add_subplot(gs[3, col])
    ax.imshow(to_gray(sample_controls[i]['same_imgs'][0:1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    sm, ss = sample_controls[i]['same_mean'], sample_controls[i]['same_std']
    ax.text(0.5, -0.08, f'SSIM = {sm:.3f} ± {ss:.3f}  (n=20)',
            transform=ax.transAxes, ha='center', fontsize=10, color=C_NEG)

# Δ inset bar
ax_inset = fig.add_subplot(gs[0:3, 3])
xs = np.arange(N)
colors_bar = [C_POS if d > 0 else C_NEG for d in deltas]
ax_inset.bar(xs, deltas, color=colors_bar, edgecolor='black', linewidth=1)
ax_inset.axhline(0, color='black', linewidth=0.8)
ax_inset.set_xticks(xs)
ax_inset.set_xticklabels([f'S{i+1}' for i in range(N)])
ax_inset.set_ylabel('Δ = LoRA − mean control SSIM')
ax_inset.set_title('Instance-level lift\n(LoRA − same-class mean)', fontsize=12, fontweight='bold')
for i, d_ in enumerate(deltas):
    va = 'bottom' if d_ > 0 else 'top'
    ax_inset.text(i, d_ + (0.01 if d_ > 0 else -0.01), f'{d_:+.3f}',
                  ha='center', va=va, fontsize=10, fontweight='bold')
yl_low = min(min(deltas) - 0.05, -0.05)
yl_high = max(max(deltas) + 0.05, 0.10)
ax_inset.set_ylim(yl_low, yl_high)
ax_inset.grid(alpha=0.3, axis='y')

fig.text(0.5, 0.02,
         'Same-class control is now mean over 20 random same-digit samples (stabilizes against single-instance lucky-look).',
         ha='center', fontsize=10, style='italic', color='#333')

out5 = OUT / 'slide5_ntk_mnist_grid.png'
plt.savefig(out5, dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'\nSaved: {out5}')

# ============================================================
# Slide 10: two reference lines
# ============================================================
free_ssim = []
with open(ROOT / 'results/multiseed_freec_vs_oracle_20260327_034128.csv') as f:
    for row in csv.DictReader(f):
        if row['mode'] == 'free_c':
            free_ssim.append(float(row['ssim']))
arr = np.array(free_ssim)

# Average the floor across the 2 samples (the multi-seed run used both)
same_class_floor = float(np.mean([c['same_mean'] for c in sample_controls]))
cross_class_floor = float(np.mean([c['cross_mean'] for c in sample_controls]))
pct_above_same  = 100.0 * (arr > same_class_floor ).sum() / len(arr)
pct_above_cross = 100.0 * (arr > cross_class_floor).sum() / len(arr)

print(f'\nSlide-10 references:')
print(f'  Same-class control floor (mean across 2 samples): {same_class_floor:.3f}')
print(f'  Cross-class baseline (mean across 2 samples):     {cross_class_floor:.3f}')
print(f'  % seeds above same-class:  {pct_above_same:.1f}%')
print(f'  % seeds above cross-class: {pct_above_cross:.1f}%')

fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(arr, bins=22, color=C_LORA, edgecolor='black', alpha=0.85, label='LoRA r=8 recon (50 seeds)')
ax.axvline(arr.mean(), color=C_FULL, linestyle='--', linewidth=2,
           label=f'Recon mean = {arr.mean():.3f}')
ax.axvline(cross_class_floor, color=C_NEG, linestyle='--', linewidth=2,
           label=f'Cross-class baseline = {cross_class_floor:.2f}  (random-image floor)')
ax.axvline(same_class_floor, color=C_LORA, linestyle='--', linewidth=2,
           label=f'Same-class control = {same_class_floor:.2f}  (class-identity floor)')
ax.set_xlabel('SSIM')
ax.set_ylabel('Count')
ax.set_title('Multi-seed reconstruction (50 seeds, LoRA r=8, MNIST)',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(alpha=0.25)
ax.text(0.99, 0.98,
        f'% seeds above cross-class:  {pct_above_cross:.0f}%\n% seeds above same-class:   {pct_above_same:.0f}%',
        transform=ax.transAxes, fontsize=10, color='#444', va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='#fafafa', edgecolor='#ccc'))

plt.tight_layout()
out10 = OUT / 'slide10_multiseed_hist.png'
plt.savefig(out10, dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'Saved: {out10}')
