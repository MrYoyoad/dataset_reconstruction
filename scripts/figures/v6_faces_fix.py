"""Re-render slide13_faces_triptych.png with proper ImageNet de-normalization.

Bug (in scripts/figures/make_v6_figures.py / .v6_refix.py):
  Face tensors are stored in ImageNet-normalized space (range ~[-2.1, 2.3]).
  Naively doing `.clamp(0, 1)` zeros most of the pixel content before SSIM,
  giving artificially low SSIM. The correct path is:
      x_pixel = (x * imagenet_std + imagenet_mean).clamp(0, 1)
  then compute SSIM on x_pixel.

This matches the project's `phase0_vit_inversion.py::compute_metrics`
exactly, and recovers the stored .pth metrics to within 1e-4.
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

C_FULL = '#1F4E79'; C_LORA = '#E07A1F'; C_NEG = '#C0392B'; C_POS = '#2C8050'

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

def denorm(t):
    return (t.float() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)

def ssim_pair(a, b):
    return kssim(a, b, window_size=3).reshape(a.shape[0], -1).mean().item()

def psnr(recon, target):
    mse = F.mse_loss(recon, target).item()
    if mse <= 0: return float('inf')
    return -10.0 * np.log10(mse + 1e-10)

def to_rgb(t):
    if t.ndim == 4: t = t[0]
    return np.clip(t.detach().cpu().permute(1, 2, 0).numpy(), 0, 1)

faces = [
    ('face1', ROOT / 'results/phase0_full_r8_n1_s42_20260428_134922_face_d3winner_freq1e-3.pth'),
    ('face2', ROOT / 'results/phase0_full_r8_n1_s42_20260513_011954_face2_d3winner_freq1e-3.pth'),
    ('face3', ROOT / 'results/phase0_full_r8_n1_s42_20260513_011954_face3_d3winner_freq1e-3.pth'),
]

gts, recons, metrics = [], [], []
for name, path in faces:
    d = torch.load(path, map_location='cpu', weights_only=False)
    xt = d['x_true']; xr = d['x_recon']
    if xt.ndim == 4 and xt.shape[0] > 1:
        xt = xt[0:1]; xr = xr[0:1]
    xt_p = denorm(xt); xr_p = denorm(xr)
    s = ssim_pair(xr_p, xt_p)
    ps = psnr(xr_p, xt_p)
    stored = d.get('metrics', {}).get('ssim')
    print(f'{name}: SSIM = {s:.4f} (stored: {stored:.4f})    PSNR = {ps:.2f} dB')
    gts.append(xt_p); recons.append(xr_p); metrics.append((s, ps))

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 10, 'figure.dpi': 150,
})

fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
for i in range(3):
    axes[0, i].imshow(to_rgb(gts[i])); axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
    for sp in axes[0, i].spines.values(): sp.set_color('#888')
    axes[0, i].set_title(f'Person {i+1}', fontsize=13)
    axes[1, i].imshow(to_rgb(recons[i])); axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
    for sp in axes[1, i].spines.values(): sp.set_color('#888')
    s, ps = metrics[i]
    axes[1, i].text(0.5, -0.08, f'SSIM = {s:.2f}    PSNR = {ps:.1f} dB',
                    transform=axes[1, i].transAxes, ha='center', fontsize=11,
                    color=C_POS, fontweight='bold')
axes[0, 0].set_ylabel('Original', fontsize=14, labelpad=10)
axes[1, 0].set_ylabel('Recovered\nfrom ViT gradient', fontsize=14, labelpad=10)
plt.subplots_adjust(wspace=0.05, hspace=0.12, top=0.93, bottom=0.06, left=0.10, right=0.99)

out = OUT / 'slide13_faces_triptych.png'
plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'\nSaved: {out}')
