"""Re-render v6 slides 5, 7, 10 with the SSIM bug fixed.

Bug (in experiments/metrics.py:compute_ssim):
  ds_mean is unconditionally added to BOTH x_recon and x_target.
  But x_train/x_ctrl are stored as already-normalized [0,1] images.
  Adding ds_mean again pushes them above 1, then .clamp damages them.

Fix:
  Add ds_mean ONLY to mean-subtracted tensors (i.e. reconstructions).
  Detect by range: if max(tensor) > 1.0 - 1e-3 and min(tensor) >= -1e-3,
  the tensor is already in [0,1]; skip ds_mean addition.
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

C_FULL = '#1F4E79'; C_LORA = '#E07A1F'; C_NEG = '#C0392B'; C_POS = '#2C8050'; C_GATE = '#7A7A7A'
GATE_SSIM = 0.30

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 10, 'figure.dpi': 150,
})

def already_in_unit_range(t):
    """True if tensor looks like a [0,1] image (not mean-subtracted)."""
    return float(t.min()) >= -0.05 and float(t.max()) <= 1.05

def maybe_add_dsmean(t, ds_mean):
    """Add ds_mean only if t looks mean-subtracted; otherwise return as-is."""
    if ds_mean is None: return t
    if already_in_unit_range(t):
        return t  # already a [0,1] image
    return t + ds_mean

def to_display(t, ds_mean):
    """Bring a tensor to [0,1] for both SSIM and visualization."""
    return maybe_add_dsmean(t, ds_mean).clamp(0, 1).float()

def ssim_pair(a, b):
    return kssim(a, b, window_size=3).reshape(a.shape[0], -1).mean().item()

def to_gray(t):
    if t.ndim == 4: t = t[0, 0]
    elif t.ndim == 3: t = t[0]
    return t.detach().cpu().numpy()

# === Slide 5 redo ===
print('='*60)
print('SLIDE 5 — bug-fixed SSIM')
print('='*60)
t5 = torch.load(ROOT / 'results/exp_b_T1_r8_free_s42_a10000.pth', map_location='cpu', weights_only=False)
xt = t5['x_train']; xrf = t5['x_recon_full']; xrl = t5['x_recon_lora']; xc = t5['x_ctrl']
dm = t5.get('ds_mean')
print(f'Ranges  x_train: [{xt.min():.3f}, {xt.max():.3f}]  x_recon_full: [{xrf.min():.3f}, {xrf.max():.3f}]  x_recon_lora: [{xrl.min():.3f}, {xrl.max():.3f}]  x_ctrl: [{xc.min():.3f}, {xc.max():.3f}]')

xt_d = to_display(xt, dm); xrf_d = to_display(xrf, dm); xrl_d = to_display(xrl, dm); xc_d = to_display(xc, dm)
print(f'Display ranges after add+clamp:  x_train: [{xt_d.min():.3f}, {xt_d.max():.3f}]  x_recon_full: [{xrf_d.min():.3f}, {xrf_d.max():.3f}]')

N = xt_d.shape[0]
full = [ssim_pair(xrf_d[i:i+1], xt_d[i:i+1]) for i in range(N)]
lora = [ssim_pair(xrl_d[i:i+1], xt_d[i:i+1]) for i in range(N)]
ctrl_single = [ssim_pair(xc_d[i:i+1], xt_d[i:i+1]) for i in range(N)]
print(f'\nFull FT SSIM per sample: {[f"{s:.3f}" for s in full]}  (stored: {t5["full_metrics"]["ssim"]:.3f})')
print(f'LoRA r=8 SSIM per sample:{[f"{s:.3f}" for s in lora]}  (stored: {t5["lora_metrics"]["ssim"]:.3f})')
print(f'Single-instance control: {[f"{s:.3f}" for s in ctrl_single]}')

# Same-class & cross-class controls (averaged over 20 MNIST test samples)
mnist = datasets.MNIST(root=str(ROOT / 'dataset_reconstruction/data'),
                      train=False, download=True, transform=transforms.ToTensor())
by_digit = {d: [] for d in range(10)}
for img, lbl in mnist: by_digit[lbl].append(img)
for d in by_digit: by_digit[d] = torch.stack(by_digit[d])

digits = [5, 0]   # Sample 1 = digit 5, Sample 2 = digit 0
rng = np.random.RandomState(2026)
sample_controls = []
for i in range(N):
    d = digits[i]
    pool_same = by_digit[d]
    pool_cross = torch.cat([by_digit[dd] for dd in range(10) if dd != d])
    idx_s = rng.choice(pool_same.shape[0], 20, replace=False)
    idx_c = rng.choice(pool_cross.shape[0], 20, replace=False)
    same = pool_same[idx_s]   # [20,1,28,28] in [0,1]
    cross = pool_cross[idx_c]
    # SSIM(LoRA recon, each control image)
    lora_i = xrl_d[i:i+1]
    same_ssims = [ssim_pair(lora_i, same[j:j+1]) for j in range(20)]
    cross_ssims = [ssim_pair(lora_i, cross[j:j+1]) for j in range(20)]
    same_arr = np.array(same_ssims); cross_arr = np.array(cross_ssims)
    print(f'\nSample {i+1} (digit {d}):')
    print(f'  Same-class control  mean={same_arr.mean():.3f} ± {same_arr.std():.3f}')
    print(f'  Cross-class baseline mean={cross_arr.mean():.3f} ± {cross_arr.std():.3f}')
    sample_controls.append({'digit': d, 'same_mean': float(same_arr.mean()),
                           'same_std': float(same_arr.std()),
                           'cross_mean': float(cross_arr.mean()),
                           'cross_std': float(cross_arr.std()),
                           'same_img0': same[0:1]})

deltas = [lora[i] - sample_controls[i]['same_mean'] for i in range(N)]
print(f'\nDeltas (LoRA SSIM − same-class control mean): {[f"{d:+.3f}" for d in deltas]}')

# Render slide 5
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(4, 4, width_ratios=[0.5, 1, 1, 1.1], height_ratios=[1,1,1,1], hspace=0.18, wspace=0.18)
labels = ['Ground Truth\n(private)', 'Full FT recon\n(1.8M params)',
          'LoRA r=8 recon\n(38K params)', 'Negative Control\n(mean over 20\nsame-class samples)']
for r, lbl in enumerate(labels):
    ax = fig.add_subplot(gs[r, 0]); ax.axis('off')
    ax.text(0.5, 0.5, lbl, ha='center', va='center', fontsize=11, fontweight='bold')
for i in range(N):
    col = i + 1
    ax = fig.add_subplot(gs[0, col]); ax.imshow(to_gray(xt_d[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    ax.set_title(f'Sample {i+1}  (digit {digits[i]})', fontsize=12)
    ax = fig.add_subplot(gs[1, col]); ax.imshow(to_gray(xrf_d[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    ax.text(0.5, -0.08, f'SSIM = {full[i]:.3f}', transform=ax.transAxes, ha='center', fontsize=11, color=C_FULL, fontweight='bold')
    ax = fig.add_subplot(gs[2, col]); ax.imshow(to_gray(xrl_d[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    ax.text(0.5, -0.08, f'SSIM = {lora[i]:.3f}', transform=ax.transAxes, ha='center', fontsize=11, color=C_LORA, fontweight='bold')
    ax = fig.add_subplot(gs[3, col]); ax.imshow(to_gray(sample_controls[i]['same_img0']), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    sm, ss = sample_controls[i]['same_mean'], sample_controls[i]['same_std']
    ax.text(0.5, -0.08, f'SSIM = {sm:.3f} ± {ss:.3f}  (n=20)', transform=ax.transAxes, ha='center', fontsize=10, color=C_NEG)

ax_in = fig.add_subplot(gs[0:3, 3])
colors_bar = [C_POS if d > 0 else C_NEG for d in deltas]
ax_in.bar(range(N), deltas, color=colors_bar, edgecolor='black', linewidth=1)
ax_in.axhline(0, color='black', linewidth=0.8)
ax_in.set_xticks(range(N)); ax_in.set_xticklabels([f'S{i+1}' for i in range(N)])
ax_in.set_ylabel('Δ = LoRA − same-class mean SSIM')
ax_in.set_title('Instance-level lift\n(LoRA − same-class mean)', fontsize=12, fontweight='bold')
for i, d_ in enumerate(deltas):
    va = 'bottom' if d_ > 0 else 'top'
    ax_in.text(i, d_ + (0.01 if d_>0 else -0.01), f'{d_:+.3f}', ha='center', va=va, fontsize=10, fontweight='bold')
ax_in.grid(alpha=0.3, axis='y')

fig.text(0.5, 0.02,
         'SSIM uses kornia window=3. ds_mean added ONLY to mean-subtracted reconstructions (fixes prior under-count).',
         ha='center', fontsize=10, style='italic', color='#333')

plt.savefig(OUT / 'slide5_ntk_mnist_grid.png', dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'Saved: {OUT / "slide5_ntk_mnist_grid.png"}')

# === Slide 7 redo ===
print('\n' + '='*60)
print('SLIDE 7 — T=10 per-sample SSIM with bug-fixed pipeline')
print('='*60)
t7 = torch.load(ROOT / 'results/exp_b_T10_full_s42_a149.pth', map_location='cpu', weights_only=False)
xt7 = t7.get('x_train', t7.get('x_true')); xr7 = t7['x_recon_full']; xc7 = t7['x_ctrl']
dm7 = t7.get('ds_mean')
xt7_d = to_display(xt7, dm7); xr7_d = to_display(xr7, dm7); xc7_d = to_display(xc7, dm7)

N7 = xt7_d.shape[0]
recon7 = [ssim_pair(xr7_d[i:i+1], xt7_d[i:i+1]) for i in range(N7)]
ctrl7 = [ssim_pair(xc7_d[i:i+1], xt7_d[i:i+1]) for i in range(N7)]
print(f'Per-sample recon SSIM: {recon7}  (stored: {t7["full_metrics"]["ssim"]:.3f})')
print(f'Per-sample control SSIM: {ctrl7}')

# Side-panel: SSIM vs T from sprint2b CSV — these were computed at run-time with the same bug,
# so they may be systematically depressed too. Note in caption.
ph2 = list(csv.DictReader(open(ROOT / 'results/sprint2b_phase2_20260223_072927.csv')))
full_by_T = {}; lora_by_T = {}
for r in ph2:
    try:
        T = int(float(r['n_steps']))
        rank = r.get('rank', '')
        if rank == '' and r['full_ssim']: full_by_T.setdefault(T, []).append(float(r['full_ssim']))
        elif rank and int(float(rank))==8 and r['lora_ssim']: lora_by_T.setdefault(T, []).append(float(r['lora_ssim']))
    except: continue
Tlist = sorted(set(full_by_T) | set(lora_by_T))
full_m = [np.mean(full_by_T[T]) for T in Tlist if T in full_by_T]
lora_m = [np.mean(lora_by_T[T]) for T in Tlist if T in lora_by_T]

fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(3, 4, width_ratios=[0.5, 1, 1, 2], hspace=0.18, wspace=0.15)
lbls = ['Ground Truth', 'Recon @ T=10\n(full FT)', 'Negative Control\n(diff. sample,\nsame class)']
for r, lbl in enumerate(lbls):
    ax = fig.add_subplot(gs[r, 0]); ax.axis('off')
    ax.text(0.5, 0.5, lbl, ha='center', va='center', fontsize=12, fontweight='bold')
for i in range(N7):
    col = i + 1
    ax = fig.add_subplot(gs[0, col]); ax.imshow(to_gray(xt7_d[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    ax.set_title(f'Sample {i+1}', fontsize=12)
    ax = fig.add_subplot(gs[1, col]); ax.imshow(to_gray(xr7_d[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    ax.text(0.5, -0.08, f'SSIM = {recon7[i]:.3f}', transform=ax.transAxes, ha='center', fontsize=11, color=C_POS, fontweight='bold')
    ax = fig.add_subplot(gs[2, col]); ax.imshow(to_gray(xc7_d[i:i+1]), cmap='gray', vmin=0, vmax=1); ax.axis('off')
    ax.text(0.5, -0.08, f'SSIM = {ctrl7[i]:.3f}', transform=ax.transAxes, ha='center', fontsize=11, color=C_NEG)

ax_side = fig.add_subplot(gs[:, 3])
if full_m:
    Tf = [T for T in Tlist if T in full_by_T]
    ax_side.plot(Tf, full_m, 'o-', color=C_FULL, linewidth=2.5, markersize=9, label='Full FT')
if lora_m:
    Tl = [T for T in Tlist if T in lora_by_T]
    ax_side.plot(Tl, lora_m, 's-', color=C_LORA, linewidth=2.5, markersize=9, label='LoRA r=8')
ax_side.set_xscale('log'); ax_side.set_xlabel('Fine-tuning steps T'); ax_side.set_ylabel('Reconstruction SSIM')
ax_side.set_ylim(0, 1.05); ax_side.grid(alpha=0.3, which='both')
ax_side.axhline(GATE_SSIM, color=C_GATE, linestyle='--', linewidth=1.2, alpha=0.7)
ax_side.text(ax_side.get_xlim()[1], GATE_SSIM + 0.01, ' gate', color=C_GATE, fontsize=9, alpha=0.8, ha='right', va='bottom')
ax_side.axvline(10, color='#bbb', linestyle=':', alpha=0.6)
ax_side.text(10.5, 0.04, 'this slide', color='#666', fontsize=9, alpha=0.8)
ax_side.legend(loc='lower left')
ax_side.set_title('Quality vs steps (LeakyReLU)')

plt.suptitle('NTK reconstruction at T = 10 (full FT, MNIST) — per-sample SSIM + quality vs T',
             fontsize=14, fontweight='bold', y=0.99)
plt.savefig(OUT / 'slide7_ntk_T10_grid.png', dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'Saved: {OUT / "slide7_ntk_T10_grid.png"}')

# === Slide 10 redo (control floors with bug-fixed SSIM) ===
print('\n' + '='*60)
print('SLIDE 10 — same/cross-class floors with bug-fixed SSIM')
print('='*60)
free_ssim = []
with open(ROOT / 'results/multiseed_freec_vs_oracle_20260327_034128.csv') as f:
    for row in csv.DictReader(f):
        if row['mode'] == 'free_c': free_ssim.append(float(row['ssim']))
arr = np.array(free_ssim)
print(f'50-seed mean (from CSV — may be depressed by same bug): {arr.mean():.3f}')

same_floor  = float(np.mean([c['same_mean']  for c in sample_controls]))
cross_floor = float(np.mean([c['cross_mean'] for c in sample_controls]))
pct_same  = 100.0 * (arr > same_floor).sum()  / len(arr)
pct_cross = 100.0 * (arr > cross_floor).sum() / len(arr)
print(f'  Same-class floor:  {same_floor:.3f}')
print(f'  Cross-class floor: {cross_floor:.3f}')
print(f'  % seeds above same:  {pct_same:.1f}%')
print(f'  % seeds above cross: {pct_cross:.1f}%')

fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(arr, bins=22, color=C_LORA, edgecolor='black', alpha=0.85, label='LoRA r=8 recon (50 seeds)')
ax.axvline(arr.mean(), color=C_FULL, linestyle='--', linewidth=2, label=f'Recon mean = {arr.mean():.3f}')
ax.axvline(cross_floor, color=C_NEG, linestyle='--', linewidth=2, label=f'Cross-class baseline = {cross_floor:.2f}  (random-image floor)')
ax.axvline(same_floor, color=C_LORA, linestyle='--', linewidth=2, label=f'Same-class control = {same_floor:.2f}  (class-identity floor)')
ax.set_xlabel('SSIM'); ax.set_ylabel('Count')
ax.set_title('Multi-seed reconstruction (50 seeds, LoRA r=8, MNIST)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(alpha=0.25)
ax.text(0.99, 0.98, f'% seeds above cross-class:  {pct_cross:.0f}%\n% seeds above same-class:   {pct_same:.0f}%',
        transform=ax.transAxes, fontsize=10, color='#444', va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='#fafafa', edgecolor='#ccc'))
plt.tight_layout()
plt.savefig(OUT / 'slide10_multiseed_hist.png', dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'Saved: {OUT / "slide10_multiseed_hist.png"}')

print('\nDone. Re-render also recomputed slide 7 SSIMs with bug fix.')
