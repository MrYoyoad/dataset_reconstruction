"""Rank-sweep visualization — REV 2 plan (notes/rank_sweep_plots_plan.md).

Headline: the multi-class q_eff reversal is a genuinely LOW-RANK effect that
attenuates with rank and vanishes at full fine-tuning (r>=N).

Reads numbers from results/jacobian_j1_ranksweep_mnist_nc{2,10}_r{R}.pth (NOT
hardcoded) + max_bce for the convergence panel from the run log (source of truth).
Anchor self-check (r=8 == bin 59 / 10cls 36, iso 0.491/0.683) is a HARD ABORT.
Fashion is read-before-cite. bsub-only (never local).
"""
import os
import re
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- STYLE_GUIDE (style_guide/plots.md + guardrails T1) -------------------
BLUE, ORANGE = "#1f77b4", "#ff7f0e"          # binary / 10-class (consistent)
RED_ALPHA = 0.08                              # out-of-spec / underfit zone
YELLOW = dict(boxstyle="round", facecolor="#fdf6c3", edgecolor="#d9c86a", alpha=0.95)
DPI = 250
plt.rcParams.update({
    "axes.labelsize": 13, "axes.titlesize": 14, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 11, "figure.titlesize": 16,
    "font.size": 12,
})
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")

RESULTS = "results"
OUTDIR = "figures/rank_sweep"
LOG = "scripts/wexac_logs/mc_rank_sweep_581629.out"
RANKS = [2, 4, 8, 16, 32]
CONVERGED = [8, 16, 32]                       # 10-class memorized (max_bce < 1e-3)
S_CANON = (320, 0.01)                         # 4*Nk adequacy point; {S,2S}-stable
EPS = [0.1, 0.3, 1.0, 3.0, 10.0]
NK = 80
PROV = ("MNIST N=10 k=8 T=1000 lr=0.5 S=320 qr seed42 | job 581629 | "
        "q_eff=q_eff|col(J), a conservative LOWER bound (>= X directions) | "
        "verified vs log (yoado-35 + yoado-30)")


def load_cell(dataset, nc, r):
    p = f"{RESULTS}/jacobian_j1_ranksweep_{dataset}_nc{nc}_r{r}.pth"
    if not os.path.exists(p):
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    cs = d["results"][S_CANON]["colspace"]
    return {
        "q_eff": {float(k): int(v) for k, v in cs["q_eff"].items()},
        "iso": float(cs["iso_ratio"]),
        "r_J": int(cs["r_J"]),
        "sigma_snr": cs["sigma_snr"].double().cpu().numpy(),
        "raw_eff_rank": float(d["raw_eff_rank"]),
    }


def parse_max_bce(log_path):
    """max_bce per (nc, rank) for mnist, from the rigor '[RIGOR] 1000' lines.
    Columns: T eff_rank hard mean_bce max_bce priv_acc held_acc memorized."""
    out = {}
    cur = None
    hdr = re.compile(r"rigor\+j1 mnist nc=(\d+) N=10 k=8 rank=(\d+)")
    rig = re.compile(r"\[RIGOR\]\s+1000\s+\S+\s+\S+\s+\S+\s+(\S+)\s+")
    if not os.path.exists(log_path):
        return out
    with open(log_path) as f:
        for line in f:
            m = hdr.search(line)
            if m:
                cur = (int(m.group(1)), int(m.group(2)))
                continue
            if cur is not None:
                r = rig.search(line)
                if r:
                    try:
                        out[cur] = float(r.group(1))
                    except ValueError:
                        pass
                    cur = None
    return out


# ---- load ------------------------------------------------------------------
BIN = {r: load_cell("mnist", 2, r) for r in RANKS}
MC = {r: load_cell("mnist", 10, r) for r in RANKS}
for r in RANKS:
    assert BIN[r] is not None and MC[r] is not None, f"missing mnist .pth at r={r}"
MAXBCE = parse_max_bce(LOG)

# ---- ANCHOR SELF-CHECK = HARD ABORT (guard against wrong-.pth / drift) -----
a_bin, a_mc = BIN[8], MC[8]
errs = []
if a_bin["q_eff"][1.0] != 59:
    errs.append(f"binary r=8 q_eff@e1={a_bin['q_eff'][1.0]} != 59")
if a_mc["q_eff"][1.0] != 36:
    errs.append(f"10-class r=8 q_eff@e1={a_mc['q_eff'][1.0]} != 36")
if abs(a_bin["iso"] - 0.491) > 0.01:
    errs.append(f"binary r=8 iso={a_bin['iso']:.3f} != 0.491")
if abs(a_mc["iso"] - 0.683) > 0.01:
    errs.append(f"10-class r=8 iso={a_mc['iso']:.3f} != 0.683")
if errs:
    sys.exit("ANCHOR SELF-CHECK FAILED (aborting, no stale figure):\n  " + "\n  ".join(errs))
print("[anchor] OK — r=8 bin 59 / 10cls 36, iso 0.491/0.683 reproduced")

os.makedirs(OUTDIR, exist_ok=True)
log2r = {r: np.log2(r) for r in RANKS}
RN_X = np.log2(10)                             # r=N=10 crossing (between 8 and 16)


def mark_rn(ax, label=True):
    ax.axvline(RN_X, color="grey", ls="--", lw=1.3, alpha=0.8)
    if label:
        ax.text(RN_X + 0.04, ax.get_ylim()[1] * 0.97, "r=N=10\nLoRA≈full-FT\n(Jang 2024)",
                fontsize=9, va="top", ha="left", color="grey")


# ============================ FIGURE 1 — headline ===========================
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("LoRA rank sweep — the multi-class q_eff reversal is a LOW-RANK effect "
             "that vanishes at full fine-tuning (r≥N)", fontweight="bold")

# -- Panel A: q_eff vs r, CONVERGED ONLY {8,16,32} --
axA = axs[0, 0]
xb = [log2r[r] for r in CONVERGED]
yb = [BIN[r]["q_eff"][1.0] / NK for r in CONVERGED]
ym = [MC[r]["q_eff"][1.0] / NK for r in CONVERGED]
axA.plot(xb, yb, "o-", color=BLUE, lw=2.4, ms=9, label="binary (nc=2)")
axA.plot(xb, ym, "s-", color=ORANGE, lw=2.4, ms=9, label="10-class (nc=10)")
for r in CONVERGED:
    axA.text(log2r[r], BIN[r]["q_eff"][1.0] / NK + 0.02, f"{BIN[r]['q_eff'][1.0]}",
             color=BLUE, fontweight="bold", fontsize=11, ha="center")
    axA.text(log2r[r], MC[r]["q_eff"][1.0] / NK - 0.045, f"{MC[r]['q_eff'][1.0]}",
             color=ORANGE, fontweight="bold", fontsize=11, ha="center")
axA.set_xticks([log2r[r] for r in CONVERGED])
axA.set_xticklabels([f"r={r}" + ("\n(r<N)" if r < 10 else "\n(r≥N)") for r in CONVERGED])
axA.set_ylabel("q_eff / 80  (recoverable fraction, ε=1)")
axA.set_title("A. q_eff vs rank — converged cells (the money panel)")
axA.set_ylim(0, 1.0)
mark_rn(axA)
axA.legend(loc="lower right")
axA.text(0.03, 0.06, "binary flat ~58–60; 10-class climbs 36→47→58 to meet it\n"
         "→ reversal shrinks as LoRA → full fine-tuning", transform=axA.transAxes,
         fontsize=10, bbox=YELLOW, va="bottom")

# -- Panel B: reversal gap vs r, converged --
axB = axs[0, 1]
gap = [BIN[r]["q_eff"][1.0] - MC[r]["q_eff"][1.0] for r in CONVERGED]
axB.plot([log2r[r] for r in CONVERGED], gap, "D-", color="#2ca02c", lw=2.6, ms=10)
for r, g in zip(CONVERGED, gap):
    axB.text(log2r[r], g + 0.7, f"{g}", color="#2ca02c", fontweight="bold",
             fontsize=12, ha="center")
axB.axhline(0, color="grey", lw=1)
axB.set_xticks([log2r[r] for r in CONVERGED])
axB.set_xticklabels([f"r={r}" for r in CONVERGED])
axB.set_ylabel("reversal gap  (binary − 10-class q_eff, ε=1)")
axB.set_title("B. The reversal closes: gap 23 → 13 → 0")
axB.set_ylim(-3, 27)
mark_rn(axB, label=False)
axB.annotate("full-FT: reversal gone", xy=(log2r[32], 0), xytext=(log2r[16], 9),
             fontsize=10, bbox=YELLOW,
             arrowprops=dict(arrowstyle="->", color="grey"))

# -- Panel C: iso vs r, both bases (mechanism decouples) --
axC = axs[1, 0]
xall = [log2r[r] for r in RANKS]
axC.plot(xall, [BIN[r]["iso"] for r in RANKS], "o-", color=BLUE, lw=2.4, ms=8,
         label="binary (nc=2)")
# 10-class: solid for converged, hollow/dashed for underfit r=2,4
axC.plot([log2r[r] for r in CONVERGED], [MC[r]["iso"] for r in CONVERGED], "s-",
         color=ORANGE, lw=2.4, ms=9, label="10-class (nc=10, converged)")
axC.plot([log2r[r] for r in (2, 4)], [MC[r]["iso"] for r in (2, 4)], "s--",
         color=ORANGE, lw=1.4, ms=9, mfc="none", alpha=0.6,
         label="10-class (underfit — see D)")
axC.set_xticks(xall)
axC.set_xticklabels([f"r={r}" for r in RANKS])
axC.set_ylabel("iso_ratio = tr(Σ_J)/(μ·r_J)   (noise-coupling)")
axC.set_title("C. Noise-coupling — mechanism DECOUPLES at r≥N")
axC.set_ylim(0, 1.0)
mark_rn(axC, label=False)
axC.legend(loc="upper left", fontsize=9)
axC.text(0.30, 0.05,
         "noise-coupling explains the reversal at r=8 (10-cls iso 0.68 > 0.49 binary);\n"
         "by r=16 the iso-gap has FLIPPED (0.39 < 0.81) yet q_eff STILL reverses →\n"
         "mechanism DECOUPLES (iso no longer explains the reversal at r≥N)",
         transform=axC.transAxes, fontsize=9.5, bbox=YELLOW, va="bottom")

# -- Panel D: convergence gate max_bce vs r (where r=2,4 live) --
axD = axs[1, 1]
have = all((10, r) in MAXBCE and (2, r) in MAXBCE for r in RANKS)
if have:
    RED = "#d62728"
    axD.axhspan(0, 1e-3, color="#2ca02c", alpha=0.06)          # "memorized" zone
    axD.axhline(1e-3, color=RED, ls="--", lw=1.6, label="memorization gate 1e-3")
    axD.semilogy(xall, [MAXBCE[(2, r)] for r in RANKS], "o-", color=BLUE, lw=2.2,
                 ms=8, label="binary (nc=2)")
    axD.semilogy(xall, [MAXBCE[(10, r)] for r in RANKS], "s-", color=ORANGE, lw=2.2,
                 ms=8, label="10-class (nc=10)")
    # shade the underfit r<8 region red (out-of-spec convention)
    axD.axvspan(log2r[2] - 0.3, (log2r[4] + log2r[8]) / 2, color="#d62728", alpha=RED_ALPHA)
    for r in (2, 4):
        axD.text(log2r[r], MAXBCE[(10, r)] * 1.25, "underfit\n(excluded)", color="#d62728",
                 fontsize=9, ha="center", va="bottom", fontweight="bold")
    axD.set_xticks(xall)
    axD.set_xticklabels([f"r={r}" for r in RANKS])
    axD.set_ylabel("max_bce at T=1000  (log scale)")
    axD.set_title("D. Convergence gate — why r=2,4 (10-class) are OFF panel A")
    axD.legend(loc="upper right", fontsize=9)
    axD.text(0.03, 0.04, "10-class r=2,4 sit ABOVE the 1e-3 memorization gate → excluded\n"
             "from the leakage panels (their huge low-r gap is an underfit artifact,\n"
             "NOT a bigger reversal). Binary memorizes at every rank.",
             transform=axD.transAxes, fontsize=9.5, bbox=YELLOW, va="bottom")
else:
    axD.text(0.5, 0.5, "max_bce not parsed from log — Panel D unavailable",
             transform=axD.transAxes, ha="center", fontsize=12, color="#d62728")

fig.text(0.5, 0.005, PROV, ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
f1 = f"{OUTDIR}/rank_sweep_headline.png"
fig.savefig(f1, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {f1}")

# ============================ FIGURE 2 — ε small multiples ==================
fig, axs = plt.subplots(1, len(CONVERGED), figsize=(16, 5.2), sharey=True)
fig.suptitle("q_eff vs ε per converged rank — the reversal lives at low ε "
             "(ε=1 in Fig 1 is representative, not cherry-picked)", fontweight="bold")
for ax, r in zip(axs, CONVERGED):
    yb = [BIN[r]["q_eff"][e] / NK for e in EPS]
    ym = [MC[r]["q_eff"][e] / NK for e in EPS]
    ax.semilogx(EPS, yb, "o-", color=BLUE, lw=2.2, ms=8, label="binary")
    ax.semilogx(EPS, ym, "s-", color=ORANGE, lw=2.2, ms=8, label="10-class")
    ax.axvline(1.0, color="grey", ls=":", lw=1.2)
    ax.set_title(f"r={r}  ({'r<N' if r < 10 else 'r≥N'})")
    ax.set_xlabel("ε (perturbation scale)")
    ax.set_xticks(EPS)
    ax.set_xticklabels([str(e) for e in EPS])
    ax.set_ylim(0, 1.02)
    if r == CONVERGED[0]:
        ax.set_ylabel("q_eff / 80")
        ax.legend(loc="upper left")
axs[0].text(0.03, 0.55, "reversal (orange<blue)\nstrongest at low ε,\nwashes out by ε≥3",
            transform=axs[0].transAxes, fontsize=9.5, bbox=YELLOW, va="top")
fig.text(0.5, 0.01, PROV, ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.03, 1, 0.92])
f2 = f"{OUTDIR}/rank_sweep_eps.png"
fig.savefig(f2, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {f2}")

# ============================ FIGURE 3 — σ-spectrum (supplementary) =========
fig, ax = plt.subplots(figsize=(10, 6))
sb = np.sort(BIN[8]["sigma_snr"])[::-1]
sm = np.sort(MC[8]["sigma_snr"])[::-1]
idx = np.arange(1, NK + 1)
ax.semilogy(idx, sb, "o-", color=BLUE, ms=4, lw=1.6, label=f"binary (q_eff@ε1={BIN[8]['q_eff'][1.0]})")
ax.semilogy(idx, sm, "s-", color=ORANGE, ms=4, lw=1.6, label=f"10-class (q_eff@ε1={MC[8]['q_eff'][1.0]})")
ax.axhline(1.0, color="#d62728", ls="--", lw=1.6, label="ε=1 threshold (σ_SNR·ε > 1 ⇒ recoverable)")
ax.set_xlabel("direction index (sorted, descending σ_SNR)")
ax.set_ylabel("σ_i(J_SNR)  — signal-to-init-noise per direction")
ax.set_title("Geometry behind the count (r=8) — 10-class has more directions "
             "below the noise floor.  NOT a reconstruction.", fontsize=12)
ax.legend(loc="upper right")
ax.text(0.03, 0.06, "q_eff = # directions above the ε=1 line.\n10-class buries more of its "
        "80 directions under the\ninit-noise floor → fewer recoverable (the reversal).",
        transform=ax.transAxes, fontsize=9.5, bbox=YELLOW, va="bottom")
fig.text(0.5, 0.005, PROV, ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.02, 1, 1])
f3 = f"{OUTDIR}/rank_sweep_spectrum_r8.png"
fig.savefig(f3, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {f3}")

# ---- Fashion: read-before-cite (no plot; rank-accurate caption) ------------
fash_bin = {r: load_cell("fashion", 2, r) for r in (8, 16)}
fash_mc = {r: load_cell("fashion", 10, r) for r in (8, 16)}
print("\n[fashion status — read from disk]")
for r in (8, 16):
    b = fash_bin[r]
    print(f"  fashion binary r={r}: " +
          (f"q_eff@ε1={b['q_eff'][1.0]}, iso={b['iso']:.3f}" if b else "NO .pth"))
for r in (8, 16):
    print(f"  fashion 10-class r={r}: " +
          ("q_eff PRESENT" if fash_mc[r] else "NO .pth (did not emit)"))
if not any(fash_mc.values()):
    print("  => Fashion 10-class q_eff UNAVAILABLE — r=8 FD-chaotic (bounded out); "
          "r=16 FD-clean+converged (max_bce 2.1e-4) but Σ_seed/q_eff did not emit. "
          "No fashion crossing plot (would mislead).")
print("\nDONE.")
