"""Re-render slide 14 (N=3 joint face reconstruction) with proper ImageNet denorm.

Same bug as the slide 13 single-face: face tensors are stored in ImageNet-
normalized space (range ~[-2.1, 2.3]). Clamping to [0,1] before SSIM/display
destroys the image and depresses SSIM. Correct path:
    x_pixel = (x * imagenet_std + imagenet_mean).clamp(0, 1)
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from kornia.metrics import ssim as kssim

ROOT = Path('/home/projects/galvardi/yoado')
OUT = ROOT / 'figures/v6'

C_POS = '#2C8050'; C_NEG = '#C0392B'; C_GATE = '#7A7A7A'
GATE_SSIM = 0.30

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

def denorm(t):
    return (t.float() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)

def ssim_pair(a, b):
    return kssim(a, b, window_size=3).reshape(a.shape[0], -1).mean().item()

def to_rgb(t):
    if t.ndim == 4: t = t[0]
    return np.clip(t.detach().cpu().permute(1, 2, 0).numpy(), 0, 1)

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 10, 'figure.dpi': 150,
})

# Load N=3 joint inversion tensor
path = ROOT / 'results/phase0_full_r8_n1_s42_20260429_005407_face_n3_same_d3winner.pth'
d = torch.load(path, map_location='cpu', weights_only=False)
print(f'Keys: {list(d.keys())}')
print(f'Stored aggregate metrics: {d.get("metrics")}')

xt = d['x_true']         # [3, 3, 224, 224] ImageNet-normalized
xr = d['x_recon']        # [3, 3, 224, 224] ImageNet-normalized
print(f'x_true  range: [{xt.min():.3f}, {xt.max():.3f}], shape: {tuple(xt.shape)}')
print(f'x_recon range: [{xr.min():.3f}, {xr.max():.3f}]')

# Apply denorm
xt_p = denorm(xt); xr_p = denorm(xr)

# Per-image SSIM (correct)
per_img_ssim = []
per_img_psnr = []
for i in range(3):
    s = ssim_pair(xr_p[i:i+1], xt_p[i:i+1])
    mse = F.mse_loss(xr_p[i:i+1], xt_p[i:i+1]).item()
    ps = -10.0 * np.log10(mse + 1e-10)
    per_img_ssim.append(s)
    per_img_psnr.append(ps)

# Cross-matrix (SSIM of recon[i] vs GT[j]) — for backup, partial-collapse visualization
cross = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        cross[i, j] = ssim_pair(xr_p[i:i+1], xt_p[j:j+1])

print('\nPer-image SSIM (bug-fixed):')
for i in range(3):
    print(f'  Person {i+1}: SSIM = {per_img_ssim[i]:.4f}  PSNR = {per_img_psnr[i]:.2f} dB')
print(f'  Mean over 3:    SSIM = {np.mean(per_img_ssim):.4f}')

print('\nCross-matrix SSIM(recon[i], GT[j]) — diagonal = self-match:')
print('             vs GT P1    vs GT P2    vs GT P3')
for i in range(3):
    row = '  '.join(f'{cross[i,j]:8.3f}' + ('*' if i == j else ' ') for j in range(3))
    print(f'  recon P{i+1}:  {row}')

# Identify centroid attraction: does recon[i] best-match GT[i]?
best_match = [int(np.argmax(cross[i])) for i in range(3)]
print(f'\nBest match per recon: {best_match} (want [0,1,2] for clean recovery)')

# === Main figure: 2 rows × 3 cols, GT vs Recon ===
fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))

for i in range(3):
    axes[0, i].imshow(to_rgb(xt_p[i:i+1])); axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
    for sp in axes[0, i].spines.values(): sp.set_color('#888')
    axes[0, i].set_title(f'Person {i+1}', fontsize=13)

    axes[1, i].imshow(to_rgb(xr_p[i:i+1])); axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
    for sp in axes[1, i].spines.values(): sp.set_color('#888')
    s = per_img_ssim[i]; ps = per_img_psnr[i]
    color = C_POS if s >= GATE_SSIM else C_NEG
    axes[1, i].text(0.5, -0.08, f'SSIM = {s:.2f}    PSNR = {ps:.1f} dB',
                    transform=axes[1, i].transAxes, ha='center', fontsize=11,
                    color=color, fontweight='bold')

axes[0, 0].set_ylabel('Original',    fontsize=14, labelpad=10)
axes[1, 0].set_ylabel('Joint N = 3\nrecovery', fontsize=14, labelpad=10)

# Subtitle line below the grid
mean_s = float(np.mean(per_img_ssim))
all_clean = all(b == i for i, b in enumerate(best_match))
note = ('All three recoveries match their own ground truth (no centroid collapse).'
        if all_clean else
        f'Partial centroid attraction: best-match for recon = {best_match} (want [0,1,2]).')
fig.text(0.5, 0.02,
         f'Per-image mean SSIM = {mean_s:.2f}.  {note}',
         ha='center', fontsize=11, style='italic', color='#333')

plt.subplots_adjust(wspace=0.05, hspace=0.12, top=0.93, bottom=0.10, left=0.10, right=0.99)
out = OUT / 'slide14_n3_joint_recovery.png'
plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'\nSaved: {out}')

# === Companion cross-matrix figure for backup ===
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(cross, cmap='YlGn', vmin=0.3, vmax=0.85, aspect='auto')
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(['GT P1', 'GT P2', 'GT P3'], fontsize=12)
ax.set_yticklabels(['Recon P1', 'Recon P2', 'Recon P3'], fontsize=12)
for i in range(3):
    for j in range(3):
        c = 'white' if cross[i,j] > 0.65 else 'black'
        weight = 'bold' if i == j else 'normal'
        ax.text(j, i, f'{cross[i,j]:.2f}', ha='center', va='center',
                color=c, fontsize=15, fontweight=weight)
ax.set_title('Cross-SSIM: recon[i] vs GT[j]  — diagonal should dominate',
             fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='SSIM')
plt.tight_layout()
out2 = OUT / 'slide14_n3_cross_matrix.png'
plt.savefig(out2, dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'Saved: {out2}')
