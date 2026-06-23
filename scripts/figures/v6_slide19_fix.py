"""Re-render slide 19 (D2 corner exemplars) cleanly.

Previous v6 embedded the pre-made d2_NN_recon.png files which contain stacked
GT+Recon images with their own internal text labels — that's why the figure
looked doubled-up and cluttered.

Fix: load each .pth tensor directly, render ONLY the reconstruction image
in each panel, add clean labels.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.metrics import ssim as kssim

ROOT = Path('/home/projects/galvardi/yoado')
OUT = ROOT / 'figures/v6'

C_FULL = '#1F4E79'; C_LORA = '#E07A1F'; C_NEG = '#C0392B'; C_POS = '#2C8050'

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

def denorm(t):
    return (t.float() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)

def to_rgb(t):
    if t.ndim == 4: t = t[0]
    return np.clip(t.detach().cpu().permute(1, 2, 0).numpy(), 0, 1)

def ssim_pair(a, b):
    return kssim(a, b, window_size=3).reshape(a.shape[0], -1).mean().item()

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 10, 'figure.dpi': 150,
})

# Four corners, identified earlier from D2 CSV
# (high_cos / high_tv) idx=35, (high_cos / low_tv) idx=03,
# (low_cos / high_tv) idx=33, (low_cos / low_tv) idx=05
corners = [
    ('High cos, low TV',  ROOT / 'results/phase0_d2_03_tv5e-03_lr0.05_it30000_20260428_063055.pth'),
    ('High cos, high TV', ROOT / 'results/phase0_d2_35_tv1e-01_lr0.05_it30000_20260428_023050.pth'),
    ('Low cos, low TV',   ROOT / 'results/phase0_d2_05_tv5e-03_lr0.1_it30000_20260428_063055.pth'),
    ('Low cos, high TV',  ROOT / 'results/phase0_d2_33_tv1e-01_lr0.01_it30000_20260428_022923.pth'),
]

# Load + extract recons + recompute SSIM
loaded = []
gt = None
for label, path in corners:
    d = torch.load(path, map_location='cpu', weights_only=False)
    xt = d.get('x_true', d.get('x_train'))
    xr = d.get('x_recon')
    if xt.ndim == 4 and xt.shape[0] > 1:
        xt = xt[0:1]; xr = xr[0:1]
    xt_p = denorm(xt); xr_p = denorm(xr)
    # SSIM
    s = ssim_pair(xr_p, xt_p)
    # cos_sim and TV stored
    metrics = d.get('metrics', {})
    cos = metrics.get('best_cos_sim', None)
    tv = d.get('tv_weight')
    print(f'{label}:  SSIM={s:.3f}  cos={cos:.3f}  TV={tv:.0e}')
    loaded.append({'label': label, 'recon': xr_p, 'ssim': s, 'cos': cos, 'tv': tv})
    if gt is None:
        gt = xt_p  # All 4 are same image

# === Layout: 1 row GT (left, slim) + 2x2 grid of recons (right) ===
# OR simpler: 2x2 of recons + a small GT panel
fig = plt.figure(figsize=(11, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 1, 1], hspace=0.25, wspace=0.1)

# Top: GT centered with caption
ax_gt = fig.add_subplot(gs[0, :])
# Center the GT image — make it a square with margins
ax_gt_in = fig.add_axes([0.42, 0.78, 0.16, 0.18])
ax_gt_in.imshow(to_rgb(gt))
ax_gt_in.set_xticks([]); ax_gt_in.set_yticks([])
for sp in ax_gt_in.spines.values(): sp.set_color('#444'); sp.set_linewidth(1.5)
ax_gt_in.set_title('Ground Truth (same target image for all 4 corners)', fontsize=11, fontweight='bold')
ax_gt.axis('off')

# 2x2 recons
positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
for entry, (r, c) in zip(loaded, positions):
    ax = fig.add_subplot(gs[r, c])
    ax.imshow(to_rgb(entry['recon']))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#888')
    # Title above the image
    ax.set_title(entry['label'], fontsize=13, fontweight='bold', pad=8)
    # SSIM / cos / TV label below
    label_text = f"SSIM = {entry['ssim']:.3f}    cos = {entry['cos']:.3f}    TV = {entry['tv']:.0e}"
    color = C_POS if entry['ssim'] >= 0.30 else C_NEG
    ax.text(0.5, -0.08, label_text, transform=ax.transAxes, ha='center', fontsize=11,
            color=color, fontweight='bold')

plt.suptitle('Same gradient match, very different reconstructions — TV weight is the lever',
             fontsize=14, fontweight='bold', y=1.0)
fig.text(0.5, 0.02,
         'All 4 share cos ≈ 0.94–0.96 (gradient match), but SSIM ranges 0.10–0.55 depending on TV weight.',
         ha='center', fontsize=11, style='italic', color='#333')

plt.subplots_adjust(top=0.93, bottom=0.06, left=0.05, right=0.95)
out = OUT / 'slide19_corner_exemplars.png'
plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'\nSaved: {out}')
