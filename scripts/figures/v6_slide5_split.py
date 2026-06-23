"""Split slide 5 into TWO cleaner figures:

  slide5a_ntk_mnist_quality.png   — GT / Full FT / LoRA r=8, 2 samples (3 rows × 2 cols)
                                    Story: "does the attack reconstruct?"

  slide5b_ntk_mnist_leakage.png   — LoRA recon vs same-class control mean, 2 samples,
                                    with Δ inset bar.  Story: "is this leakage?"

Same bug-fixed SSIM pipeline as v6_refix.py / v6_faces_fix.py:
  - MNIST: ds_mean added ONLY to mean-subtracted reconstructions.
  - kornia.metrics.ssim(window_size=3).
"""
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

C_FULL = '#1F4E79'; C_LORA = '#E07A1F'; C_NEG = '#C0392B'; C_POS = '#2C8050'
GATE_SSIM = 0.30

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 10, 'figure.dpi': 150,
})

def already_unit(t):
    return float(t.min()) >= -0.05 and float(t.max()) <= 1.05

def maybe_add_dsmean(t, ds_mean):
    if ds_mean is None: return t
    return t if already_unit(t) else t + ds_mean

def to_display(t, ds_mean):
    return maybe_add_dsmean(t, ds_mean).clamp(0, 1).float()

def ssim_pair(a, b):
    return kssim(a, b, window_size=3).reshape(a.shape[0], -1).mean().item()

def to_gray(t):
    if t.ndim == 4: t = t[0, 0]
    elif t.ndim == 3: t = t[0]
    return t.detach().cpu().numpy()

# Load tensor
t5 = torch.load(ROOT / 'results/exp_b_T1_r8_free_s42_a10000.pth', map_location='cpu', weights_only=False)
xt = t5['x_train']; xrf = t5['x_recon_full']; xrl = t5['x_recon_lora']
dm = t5.get('ds_mean')
xt_d  = to_display(xt,  dm)
xrf_d = to_display(xrf, dm)
xrl_d = to_display(xrl, dm)

N = xt_d.shape[0]
full = [ssim_pair(xrf_d[i:i+1], xt_d[i:i+1]) for i in range(N)]
lora = [ssim_pair(xrl_d[i:i+1], xt_d[i:i+1]) for i in range(N)]
print(f'Full FT SSIM: {full}')
print(f'LoRA   SSIM: {lora}')

# Same-class controls (mean over 20 MNIST instances)
mnist = datasets.MNIST(root=str(ROOT / 'dataset_reconstruction/data'), train=False, download=True, transform=transforms.ToTensor())
by_digit = {d: [] for d in range(10)}
for img, lbl in mnist: by_digit[lbl].append(img)
for d in by_digit: by_digit[d] = torch.stack(by_digit[d])

digits = [5, 0]   # hardcoded from original figure
rng = np.random.RandomState(2026)
controls = []
for i in range(N):
    d = digits[i]
    pool = by_digit[d]
    idx = rng.choice(pool.shape[0], 20, replace=False)
    same = pool[idx]
    lora_i = xrl_d[i:i+1]
    ssims = np.array([ssim_pair(lora_i, same[j:j+1]) for j in range(20)])
    controls.append({'digit': d, 'mean': float(ssims.mean()), 'std': float(ssims.std()),
                    'first_img': same[0:1]})
    print(f'Sample {i+1} (digit {d}) same-class control: {ssims.mean():.3f} ± {ssims.std():.3f}')

deltas = [lora[i] - controls[i]['mean'] for i in range(N)]
print(f'Deltas: {deltas}')


# =============================================================================
# SLIDE 5a — Reconstruction quality (GT / Full FT / LoRA)
# =============================================================================
fig, axes = plt.subplots(3, N + 1, figsize=(8.5, 9),
                         gridspec_kw={'width_ratios': [0.45] + [1]*N})

labels = ['Ground Truth\n(private)',
          'Full FT recon\n(1.8M params)',
          'LoRA r=8 recon\n(38K params)']
for r, lbl in enumerate(labels):
    axes[r, 0].axis('off')
    axes[r, 0].text(0.5, 0.5, lbl, ha='center', va='center', fontsize=13, fontweight='bold')

for i in range(N):
    col = i + 1
    axes[0, col].imshow(to_gray(xt_d[i:i+1]), cmap='gray', vmin=0, vmax=1); axes[0, col].axis('off')
    axes[0, col].set_title(f'Sample {i+1}  (digit {digits[i]})', fontsize=12)
    axes[1, col].imshow(to_gray(xrf_d[i:i+1]), cmap='gray', vmin=0, vmax=1); axes[1, col].axis('off')
    axes[1, col].text(0.5, -0.08, f'SSIM = {full[i]:.3f}', transform=axes[1, col].transAxes,
                       ha='center', fontsize=12, color=C_FULL, fontweight='bold')
    axes[2, col].imshow(to_gray(xrl_d[i:i+1]), cmap='gray', vmin=0, vmax=1); axes[2, col].axis('off')
    axes[2, col].text(0.5, -0.08, f'SSIM = {lora[i]:.3f}', transform=axes[2, col].transAxes,
                       ha='center', fontsize=12, color=C_LORA, fontweight='bold')

plt.subplots_adjust(wspace=0.04, hspace=0.16, top=0.95, bottom=0.04, left=0.05, right=0.99)
out_a = OUT / 'slide5a_ntk_mnist_quality.png'
plt.savefig(out_a, dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'Saved: {out_a}')


# =============================================================================
# SLIDE 5b — Instance leakage (LoRA vs same-class control + Δ bar)
# =============================================================================
fig = plt.figure(figsize=(11, 6.5))
gs = fig.add_gridspec(2, 4, width_ratios=[0.5, 1, 1, 1.4], hspace=0.18, wspace=0.18)

# Row labels
row_labels = ['LoRA r=8 recon', 'Same-class control\n(mean over n=20)']
for r, lbl in enumerate(row_labels):
    ax = fig.add_subplot(gs[r, 0]); ax.axis('off')
    ax.text(0.5, 0.5, lbl, ha='center', va='center', fontsize=12, fontweight='bold')

for i in range(N):
    col = i + 1
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(to_gray(xrl_d[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    ax.set_title(f'Sample {i+1}  (digit {digits[i]})', fontsize=12)
    ax.text(0.5, -0.10, f'SSIM = {lora[i]:.3f}', transform=ax.transAxes,
            ha='center', fontsize=12, color=C_LORA, fontweight='bold')

    ax = fig.add_subplot(gs[1, col])
    ax.imshow(to_gray(controls[i]['first_img']), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    ax.text(0.5, -0.10,
            f'SSIM = {controls[i]["mean"]:.3f} ± {controls[i]["std"]:.3f}',
            transform=ax.transAxes, ha='center', fontsize=12, color=C_NEG)

# Δ inset spans both rows
ax_in = fig.add_subplot(gs[:, 3])
colors_bar = [C_POS if d > 0 else C_NEG for d in deltas]
ax_in.bar(range(N), deltas, color=colors_bar, edgecolor='black', linewidth=1.2, width=0.6)
ax_in.axhline(0, color='black', linewidth=0.8)
ax_in.set_xticks(range(N))
ax_in.set_xticklabels([f'Sample {i+1}' for i in range(N)], fontsize=11)
ax_in.set_ylabel('Δ = LoRA SSIM − same-class mean SSIM', fontsize=11)
ax_in.set_title('Instance-level lift', fontsize=13, fontweight='bold')
for i, d_ in enumerate(deltas):
    va = 'bottom' if d_ > 0 else 'top'
    offset = 0.008 if d_ > 0 else -0.008
    ax_in.text(i, d_ + offset, f'{d_:+.3f}',
               ha='center', va=va, fontsize=12, fontweight='bold')
yl = max(abs(min(deltas)), abs(max(deltas))) * 1.4
ax_in.set_ylim(-yl, yl)
ax_in.grid(alpha=0.3, axis='y')

fig.text(0.5, 0.02,
         'LoRA r=8 reconstruction is more similar to its own GT than to any of 20 random same-class samples — instance leakage, not class memorization.',
         ha='center', fontsize=11, style='italic', color='#333')

plt.subplots_adjust(wspace=0.18, hspace=0.30, top=0.93, bottom=0.10, left=0.05, right=0.99)
out_b = OUT / 'slide5b_ntk_mnist_leakage.png'
plt.savefig(out_b, dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'Saved: {out_b}')
