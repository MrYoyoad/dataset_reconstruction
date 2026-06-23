"""Re-render slide 8 with the gate annotation centered and the figure decluttered.

Issues with previous v6:
  - "gate" label in top-right corner, hidden by the legend
  - "available T" note box in top-left, distracting
  - Trend lines extended off the plot
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
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 10, 'figure.dpi': 150,
})

# Load
ph0 = list(csv.DictReader(open(ROOT / 'results/sprint2b_phase0_20260222_191555.csv')))
ph2 = list(csv.DictReader(open(ROOT / 'results/sprint2b_phase2_20260223_072927.csv')))

acts = {'leaky_relu': C_POS, 'modified_relu': C_LORA, 'relu_default': C_NEG}
labels_act = {'leaky_relu': 'LeakyReLU', 'modified_relu': 'Modified ReLU', 'relu_default': 'ReLU'}

data = []
for r in ph0:
    a = r.get('activation', '')
    if a not in acts: continue
    try:
        le = float(r['full_linearization_error'])
        ss_str = r.get('full_ssim','')
        if not ss_str or ss_str == 'nan': continue
        ss = float(ss_str)
        data.append((max(le, 1e-15), ss, a))
    except (ValueError, KeyError): continue
for r in ph2:
    if r.get('rank','') != '': continue
    try:
        le = float(r['full_linearization_error'])
        ss = float(r['full_ssim'])
        data.append((max(le, 1e-15), ss, 'leaky_relu'))
    except (ValueError, KeyError): continue

print(f'Total points: {len(data)}')

fig, ax = plt.subplots(figsize=(11, 6.5))

# X-axis range: clip to actual data range to avoid long empty tails
all_le = [d[0] for d in data]
xmin, xmax = min(all_le), max(all_le)
# Use a margin in log space
xmin_plot = 10 ** (np.log10(xmin) - 0.5)
xmax_plot = 10 ** (np.log10(xmax) + 0.5)

for a, col in acts.items():
    pts = [(le, ss) for le, ss, act in data if act == a]
    if not pts: continue
    les = np.array([p[0] for p in pts])
    sss = np.array([p[1] for p in pts])
    ax.scatter(les, sss, c=col, s=110, alpha=0.75, edgecolor='black', linewidth=0.8, label=labels_act[a], zorder=3)

    # Polynomial trend — clip to actual data range to avoid off-plot extrapolation
    log_les = np.log10(les)
    if len(log_les) >= 3 and np.ptp(log_les) > 0.5:
        try:
            coeffs = np.polyfit(log_les, sss, 2)
            log_min, log_max = log_les.min(), log_les.max()
            xx = np.linspace(log_min, log_max, 50)
            yy = np.clip(np.polyval(coeffs, xx), 0, 1.05)
            ax.plot(10**xx, yy, color=col, linewidth=1.4, alpha=0.55, linestyle='-', zorder=2)
        except Exception:
            pass

# --- Gate annotation: centered text ABOVE the line, in the middle of the plot ---
ax.axhline(GATE_SSIM, color=C_GATE, linestyle='--', linewidth=1.5, alpha=0.85, zorder=1)
# Use a centered text, well-placed
# Place at the geometric center of the visible x-range (log scale)
mid_x = 10 ** ((np.log10(xmin_plot) + np.log10(xmax_plot)) / 2)
ax.text(mid_x, GATE_SSIM + 0.02, 'recognizability gate  (SSIM = 0.30)',
        color=C_GATE, fontsize=11, fontweight='bold',
        ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_GATE, alpha=0.85))

ax.set_xscale('log')
ax.set_xlim(xmin_plot, xmax_plot)
ax.set_ylim(0, 1.05)
ax.set_xlabel('Linearization error  (log scale; lower = NTK approximation more accurate)')
ax.set_ylabel('Reconstruction SSIM')
ax.set_title('NTK stability ↔ reconstruction quality   (across T, 3 activations)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, which='both')
ax.legend(loc='lower left', framealpha=0.95)

plt.tight_layout()
out = OUT / 'slide8_stability_vs_quality.png'
plt.savefig(out, dpi=180, facecolor='white', bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
