"""Slide 8 v2 — use T on x-axis (continuous) instead of linearization error (bimodal).

The previous version had a bimodal x-axis: T=1 had linearization_error ≈ 0
(numerical zero ~10⁻¹⁵), while T≥5 had values in the 10⁰–10² range. The
empty stretch between 10⁻¹³ and 10⁻¹ looked ugly, and the polynomial fit
over only-two-clusters extrapolated wildly off the top of the chart.

T is a continuous quantity that directly governs stability decay — same
story, cleaner plot.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/projects/galvardi/yoado')
OUT = ROOT / 'figures/v6'

C_FULL = '#1F4E79'; C_LORA = '#E07A1F'; C_NEG = '#C0392B'; C_POS = '#2C8050'; C_GATE = '#7A7A7A'
GATE_SSIM = 0.30

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 11, 'figure.dpi': 150,
})

ph0 = list(csv.DictReader(open(ROOT / 'results/sprint2b_phase0_20260222_191555.csv')))
ph2 = list(csv.DictReader(open(ROOT / 'results/sprint2b_phase2_20260223_072927.csv')))

acts = ['leaky_relu', 'modified_relu', 'relu_default']
colors = {'leaky_relu': C_POS, 'modified_relu': C_LORA, 'relu_default': C_NEG}
labels = {'leaky_relu': 'LeakyReLU', 'modified_relu': 'Modified ReLU', 'relu_default': 'ReLU'}

data = {a: [] for a in acts}   # T -> list of SSIMs
for r in ph0:
    a = r.get('activation', '')
    if a not in acts: continue
    try:
        T = int(float(r['n_steps']))
        ss = r.get('full_ssim','')
        if ss and ss != 'nan': data[a].append((T, float(ss)))
    except: pass
for r in ph2:
    if r.get('rank','') != '': continue
    try:
        T = int(float(r['n_steps'])); ss = float(r['full_ssim'])
        data['leaky_relu'].append((T, ss))
    except: pass

# Aggregate to (T, mean SSIM, std SSIM) per activation
agg = {a: {} for a in acts}
for a in acts:
    for T, ss in data[a]:
        agg[a].setdefault(T, []).append(ss)
    agg[a] = {T: (np.mean(v), np.std(v) if len(v)>1 else 0.0) for T, v in agg[a].items()}

for a in acts:
    print(f'{a}:  {[(T, round(agg[a][T][0], 3)) for T in sorted(agg[a])]}')

# === Plot ===
fig, ax = plt.subplots(figsize=(10.5, 6.2))

Tmin, Tmax = 1, 100
for a in acts:
    Ts = sorted(agg[a].keys())
    means = [agg[a][T][0] for T in Ts]
    stds  = [agg[a][T][1] for T in Ts]
    ax.errorbar(Ts, means, yerr=stds, fmt='o-', color=colors[a], linewidth=2.5,
                markersize=11, label=labels[a], capsize=4, capthick=1.5, alpha=0.95)

ax.set_xscale('log')
ax.set_xlim(0.8, 130)
ax.set_xlabel('Fine-tuning steps  T  (log scale)', labelpad=8)
ax.set_ylabel('Reconstruction SSIM')
# Give headroom above 1.0 so T=1 markers (SSIM=1.0) sit ≈90% up the plot, not flush with the top.
ax.set_ylim(0.40, 1.10)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.axhline(1.0, color='#cccccc', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)
ax.text(0.85, 1.005, 'SSIM = 1.0  (perfect)', color='#888', fontsize=9, va='bottom', ha='left')

# Gate line dropped from this slide: all data points are well above SSIM=0.30,
# and at y_min=0.40 the gate annotation rendered outside the plot area and
# collided with the x-axis label. The gate's pass/fail framing isn't useful
# here — the story is the activation-dependent decay rate, not absolute SSIM.

# Annotations: ReLU collapse vs LeakyReLU stability
ax.annotate('ReLU collapses\nat T ≥ 50',
            xy=(50, 0.495), xytext=(7, 0.55),
            fontsize=11, color=C_NEG, ha='left', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_NEG, alpha=0.8, lw=1.5))
ax.annotate('LeakyReLU\nstable through T=100',
            xy=(100, 0.80), xytext=(20, 0.96),
            fontsize=11, color=C_POS, ha='left', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_POS, alpha=0.8, lw=1.5))

ax.set_title('NTK approximation stability across activations  —  reconstruction SSIM vs T',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, which='both')
ax.legend(loc='lower left', framealpha=0.95)

plt.tight_layout()
out = OUT / 'slide8_stability_vs_quality.png'
plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'\nSaved: {out}')
