"""Slide 8 v3 — linearization error on x (per brief), T=1 anchor dropped.

Why T=1 is dropped from the scatter:
  Linearization error at T=1 is 0 (perfect by construction).
  On a log scale, 0 maps to -∞; matplotlib clips it to ~10⁻¹⁵, which is
  ~13 orders of magnitude below any other data point. That single cluster
  pulled polynomial fits wildly off-screen. The T=1 points carry no
  information about stability decay (it's the anchor by definition), so
  we exclude them from the visual but annotate the omission.

The visible x-range therefore starts at the actual T≥5 linearization-error
values (~4 to ~98), spanning ~10⁰ to ~10² on log scale.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/projects/galvardi/yoado')
OUT  = ROOT / 'figures/v6'

C_FULL = '#1F4E79'; C_LORA = '#E07A1F'; C_NEG = '#C0392B'; C_POS = '#2C8050'; C_GATE = '#7A7A7A'

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 11, 'figure.dpi': 150,
})

ph0 = list(csv.DictReader(open(ROOT / 'results/sprint2b_phase0_20260222_191555.csv')))
ph2 = list(csv.DictReader(open(ROOT / 'results/sprint2b_phase2_20260223_072927.csv')))

acts = {'leaky_relu': C_POS, 'modified_relu': C_LORA, 'relu_default': C_NEG}
labels = {'leaky_relu': 'LeakyReLU', 'modified_relu': 'Modified ReLU', 'relu_default': 'ReLU'}

data = []  # (lin_err, ssim, activation, T)
n_dropped_T1 = 0
for r in ph0:
    a = r.get('activation', '')
    if a not in acts: continue
    try:
        le = float(r['full_linearization_error'])
        T = int(float(r['n_steps']))
        ss = r.get('full_ssim','')
        if not ss or ss == 'nan': continue
        if T == 1:
            n_dropped_T1 += 1
            continue
        data.append((le, float(ss), a, T))
    except (ValueError, KeyError): continue

for r in ph2:
    if r.get('rank','') != '': continue
    try:
        le = float(r['full_linearization_error'])
        T = int(float(r['n_steps']))
        ss = float(r['full_ssim'])
        if T == 1:
            n_dropped_T1 += 1
            continue
        data.append((le, ss, 'leaky_relu', T))
    except (ValueError, KeyError): continue

print(f'Points kept (T ≥ 5): {len(data)}; T=1 anchor points dropped: {n_dropped_T1}')

# Plot
fig, ax = plt.subplots(figsize=(10.5, 6.2))

xmin_data = min(p[0] for p in data)
xmax_data = max(p[0] for p in data)
xmin_plot = 10 ** (np.log10(xmin_data) - 0.4)
xmax_plot = 10 ** (np.log10(xmax_data) + 0.4)

for a, col in acts.items():
    pts = sorted([(le, ss, T) for le, ss, act, T in data if act == a], key=lambda x: x[0])
    if not pts: continue
    les = np.array([p[0] for p in pts])
    sss = np.array([p[1] for p in pts])
    ax.scatter(les, sss, c=col, s=130, alpha=0.85, edgecolor='black', linewidth=0.9,
               label=labels[a], zorder=3)
    # Trend: polynomial degree 2 on log(le)
    if len(les) >= 3:
        log_les = np.log10(les)
        coeffs = np.polyfit(log_les, sss, 2)
        xx = np.linspace(log_les.min(), log_les.max(), 60)
        yy = np.clip(np.polyval(coeffs, xx), 0, 1.0)
        ax.plot(10**xx, yy, color=col, linewidth=1.6, alpha=0.55, zorder=2)

ax.set_xscale('log')
ax.set_xlim(xmin_plot, xmax_plot)
ax.set_xlabel('Linearization error  (log scale; lower = NTK approximation more accurate)',
              labelpad=8)
ax.set_ylabel('Reconstruction SSIM')

# Y range: data is in [0.49, 0.81]. Give a bit of headroom each side, place SSIM=1 marker.
ax.set_ylim(0.40, 1.10)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.axhline(1.0, color='#cccccc', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)
ax.text(xmin_plot * 1.05, 1.005, 'SSIM = 1.0  (perfect, T = 1 anchor)',
        color='#888', fontsize=9, va='bottom', ha='left')

# Annotations: activation differences
relu_pt = max([(le, ss) for le, ss, a, T in data if a == 'relu_default'], key=lambda x: x[0])
ax.annotate('ReLU collapses\nat high lin-err\n(T = 50)',
            xy=relu_pt, xytext=(relu_pt[0] / 8, 0.55),
            fontsize=11, color=C_NEG, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_NEG, alpha=0.8, lw=1.5))
lr_pt = max([(le, ss) for le, ss, a, T in data if a == 'leaky_relu'], key=lambda x: x[0])
ax.annotate('LeakyReLU stays\nrobust at\nlin-err ≈ 10²',
            xy=lr_pt, xytext=(lr_pt[0] / 6, 0.95),
            fontsize=11, color=C_POS, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_POS, alpha=0.8, lw=1.5))

# Footnote: T=1 anchor disclosure
fig.text(0.5, -0.02,
         f'T = 1 anchor not shown ({n_dropped_T1} points at linearization error ≈ 0, '
         'SSIM ≈ 1.0 by construction — outside log-scale range).',
         ha='center', fontsize=9, color='#888', style='italic')

ax.set_title('NTK stability ↔ reconstruction quality   (across 3 activations)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, which='both')
ax.legend(loc='lower left', framealpha=0.95)

plt.tight_layout()
out = OUT / 'slide8_stability_vs_quality.png'
plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
