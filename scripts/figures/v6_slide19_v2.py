"""Slide 19 v2 — clean horizontal layout.

v1's 2x2 grid + footer text was cramped, and the per-panel SSIM/cos/TV line
rendered with stretched letter-spacing (likely a bbox_inches='tight' + tiny
figure side effect). Redo as a single wide row: GT + 4 recons, big readable
text under each panel.

Layout:
  [ GT ] [ HC, low TV ] [ HC, high TV ] [ LC, low TV ] [ LC, high TV ]

Each recon panel: bold title above (corner name), SSIM (big, color-coded) on
one line below, (cos, TV) on a small second line. No more squished text.
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

C_NEG = '#C0392B'; C_POS = '#2C8050'

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
    'font.family': 'DejaVu Sans', 'figure.dpi': 150,
})

corners = [
    ('High cos, low TV',  ROOT / 'results/phase0_d2_03_tv5e-03_lr0.05_it30000_20260428_063055.pth'),
    ('High cos, high TV', ROOT / 'results/phase0_d2_35_tv1e-01_lr0.05_it30000_20260428_023050.pth'),
    ('Low cos, low TV',   ROOT / 'results/phase0_d2_05_tv5e-03_lr0.1_it30000_20260428_063055.pth'),
    ('Low cos, high TV',  ROOT / 'results/phase0_d2_33_tv1e-01_lr0.01_it30000_20260428_022923.pth'),
]

loaded = []
gt = None
for label, path in corners:
    d = torch.load(path, map_location='cpu', weights_only=False)
    xt = d.get('x_true', d.get('x_train'))
    xr = d.get('x_recon')
    if xt.ndim == 4 and xt.shape[0] > 1:
        xt = xt[0:1]; xr = xr[0:1]
    xt_p = denorm(xt); xr_p = denorm(xr)
    s = ssim_pair(xr_p, xt_p)
    metrics = d.get('metrics', {})
    cos = metrics.get('best_cos_sim', None)
    tv = d.get('tv_weight')
    print(f'{label}:  SSIM={s:.3f}  cos={cos:.3f}  TV={tv:.0e}')
    loaded.append({'label': label, 'recon': xr_p, 'ssim': s, 'cos': cos, 'tv': tv})
    if gt is None: gt = xt_p

# === Wide single-row layout ===
fig, axes = plt.subplots(1, 5, figsize=(20, 6.2))
plt.subplots_adjust(top=0.82, bottom=0.18, left=0.02, right=0.98, wspace=0.08)

# Panel 0 — GT
ax = axes[0]
ax.imshow(to_rgb(gt))
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_color('#333'); sp.set_linewidth(2.0)
ax.set_title('Ground Truth', fontsize=15, fontweight='bold', pad=10, color='#333')
ax.text(0.5, -0.06, '(same target\nfor all 4)',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=11, color='#666', style='italic')

# Panels 1–4 — recons
for i, entry in enumerate(loaded):
    ax = axes[i + 1]
    ax.imshow(to_rgb(entry['recon']))
    ax.set_xticks([]); ax.set_yticks([])
    color = C_POS if entry['ssim'] >= 0.30 else C_NEG
    for sp in ax.spines.values():
        sp.set_color(color); sp.set_linewidth(2.0)
    ax.set_title(entry['label'], fontsize=15, fontweight='bold', pad=10, color='#222')
    # Big SSIM line
    ax.text(0.5, -0.06, f"SSIM = {entry['ssim']:.3f}",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=15, fontweight='bold', color=color)
    # Smaller secondary line
    ax.text(0.5, -0.16,
            f"cos = {entry['cos']:.3f}     TV = {entry['tv']:.0e}",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=11, color='#555')

fig.suptitle('Same gradient match, very different reconstructions  —  TV weight is the lever',
             fontsize=17, fontweight='bold', y=0.96)
fig.text(0.5, 0.04,
         'All 4 share cos ≈ 0.94–0.96 (gradient match), but SSIM ranges 0.10–0.55 depending on TV weight.',
         ha='center', fontsize=12, style='italic', color='#333')

out = OUT / 'slide19_corner_exemplars.png'
plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'\nSaved: {out}')
