"""Clean, slide-native figures for the 2026-08-31 supervisor deck.

One message per panel, presentation-scale fonts, no footers / annotation prose (that lives in the
speaker notes). Every panel reads the SAME data files as the existing analysis generators:
  crux ladder      results/rescored_freec_ladder_2026-08-29.csv (+ oracle rescored_activations_857271_full_2026-08-28.csv)
  fs-vs-T          results/rescored_tsweep_2026-08-29.csv
  anchor           results/anchor_sweep_T10_r8_{relu,softplus}_s42.pth
  rank sweep       results/jacobian_j1_ranksweep_mnist_nc{2,10}_r{8,16,32}.pth   (job 581629)
  battery          results/arm_{b,c,d,e}_*/*.json, results/vit_lora_sensitivity/*.pth
  g0 / ladder / H  results/margin_at_scale/summary.json, results/similarity_ladder/ladder_t*.pth, results/h_spotcheck/h_spotcheck.json
  full-FT          results/fullft_valley/{F_summary,valley_headline_dstar}.json
  atlas            results/atlas_zoo/zoo_bank.pth  (via experiments.dataset_sensitivity.atlas_analyze)
  gallery          results/gb_e2e_{mnist,cifar10,flowers32}_N2_gelu.pth
Usage:  python scripts/deck/make_deck_figures.py [--only name,name]
Output: figures/deck_2026_08_31/*.png   (CPU only; no bsub)
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from deck import config as C  # noqa: E402

RES = C.RESULTS
OUT = C.FIG_DIR
os.makedirs(OUT, exist_ok=True)

BLUE, RED, GREEN, GRAY, AMBER, ORANGE = C.HEX_BLUE, C.HEX_RED, C.HEX_GREEN, C.HEX_GRAY, C.HEX_AMBER, C.HEX_ORANGE
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100,
})
DPI = 200


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] {p}")
    return p


def _f(r, k):
    try:
        return float(r[k])
    except Exception:
        return np.nan


# ============================================================================ crux (Part 1)
SPECTRUM = ["relu", "leaky_relu", "hardswish", "elu", "celu", "selu", "tanh", "sigmoid",
            "gelu", "gelu_tanh", "mish", "silu", "softplus"]          # kinked -> smooth (plot_freec_ladder.py)
KINKED = {"relu", "leaky_relu", "hardswish", "selu"}
RUNGS = [0.005, 0.03, 0.1, 0.3]


def _act(fname, free):
    base = os.path.basename(fname)
    if free:
        m = re.match(r"exp_b_T1_r8_free_s42_a149_([a-z_0-9.]+?)_lr[0-9.]+\.pth$", base)
    else:
        m = re.match(r"exp_b_T1_r8_s42_a149_([a-z_0-9.]+?)(?:_lr[0-9.]+)?\.pth$", base)
    if not m:
        return None
    tag = m.group(1)
    if any(s in tag for s in ("_npc", "_vw", "pbox")):
        return None
    return tag


def _by_activation_rung(csv_path, free):
    out = {}
    for r in csv.DictReader(open(csv_path)):
        act = _act(r["file"], free)
        if act is None or act not in SPECTRUM:
            continue
        wc = _f(r, "weight_change")
        rung = min(RUNGS, key=lambda g: abs(wc - g))
        rec = (_f(r, "ctrl_margin_norm"), wc, _f(r, "feature_stability"))
        d = out.setdefault(act, {})
        if rung not in d or abs(wc - rung) < abs(d[rung][1] - rung):
            d[rung] = rec
    return out


def fig_crux_bars(rung=0.1):
    freec = _by_activation_rung(f"{RES}/rescored_freec_ladder_2026-08-29.csv", True)
    oracle = _by_activation_rung(f"{RES}/rescored_activations_857271_full_2026-08-28.csv", False)
    acts = [a for a in SPECTRUM if a in freec and rung in freec[a]]
    lk = [freec[a][rung][0] for a in acts]
    cols = [RED if a in KINKED else GREEN for a in acts]
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    x = np.arange(len(acts))
    ax.bar(x, lk, color=cols, width=0.68, label=None)
    ox = [i for i, a in enumerate(acts) if a in oracle and rung in oracle[a]]
    ax.scatter(ox, [oracle[acts[i]][rung][0] for i in ox], marker="D", s=70, color=BLUE, zorder=5,
               label="oracle coefficients (upper bound)")
    ax.scatter([], [], marker="s", s=90, color=RED, label="kinked  (relu / leaky-relu / hardswish / selu)")
    ax.scatter([], [], marker="s", s=90, color=GREEN, label="smooth")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "-") for a in acts], rotation=35, ha="right")
    ax.set_ylabel("leakage  (control-margin)")
    ax.set_xlabel("activation, ordered kinked → smooth")
    ax.legend(loc="upper right", frameon=False)
    kin = np.mean([l for a, l in zip(acts, lk) if a in KINKED and a != "hardswish"])
    smo = np.mean([l for a, l in zip(acts, lk) if a not in KINKED])
    print(f"[crux] rung {rung}: kinked(relu/leaky/selu) mean {kin:.2f} vs smooth mean {smo:.2f}  ratio {kin/smo:.1f}x")
    return save(fig, "crux_bars.png")


def fig_fs_vs_T():
    rows = list(csv.DictReader(open(f"{RES}/rescored_tsweep_2026-08-29.csv")))
    series = {}
    for r in rows:
        m = re.match(r"exp_b_T(\d+)_r8_s42_a149_([a-z_0-9.]+)\.pth$", os.path.basename(r["file"]))
        if not m or any(s in m.group(2) for s in ("_lr", "_npc", "_vw", "free", "pbox")):
            continue
        series.setdefault(m.group(2), {})[int(m.group(1))] = _f(r, "feature_stability")
    show = [("softplus", GREEN, "-"), ("sigmoid", "#4FA36B", "-"), ("gelu", BLUE, "-"),
            ("leaky_relu", "#E0776A", "--"), ("relu", RED, "--")]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.axhline(0.99, color=GRAY, ls=":", lw=1.5)
    ax.text(1.05, 0.992, "linear regime (0.99)", color=GRAY, fontsize=12, va="bottom")
    for a, c, ls in show:
        if a not in series:
            continue
        Ts = sorted(series[a])
        ax.plot(Ts, [series[a][t] for t in Ts], ls, marker="o", color=c, lw=2.6, ms=7, label=a.replace("_", "-"))
    ax.set_xscale("log")
    ax.set_xlabel("fine-tuning steps  T")
    ax.set_ylabel("feature stability   cos(∇f(θ₀), ∇f(θ_T))")
    ax.set_ylim(0.45, 1.02)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return save(fig, "fs_vs_T.png")


def fig_anchor():
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax2 = ax.twinx()
    ax2.grid(False)
    for act, c in (("relu", RED), ("softplus", GREEN)):
        a = torch.load(f"{RES}/anchor_sweep_T10_r8_{act}_s42.pth", map_location="cpu", weights_only=False)
        al = a["curves"]["alphas"]
        lin = a["curves"]["lora_lin_fs"]
        cm = [a["per_alpha"][x]["lora_metrics"]["ssim_norm"] - a["per_alpha"][x]["control_metrics"]["ssim_norm"]
              for x in al]
        print(f"[anchor] {act}: ctrl_margin(α)={[round(v,3) for v in cm]}  lin_fs={[round(v,3) for v in lin]}")
        ax.plot(al, cm, "-", marker="s", color=c, lw=2.6, ms=8, label=f"{act}: leakage")
        ax2.plot(al, lin, "--", marker="o", color=c, lw=2.0, ms=6, alpha=0.8, label=f"{act}: linearization error")
    ax.set_xlabel("anchor α    θ(α) = (1−α)·θ₀ + α·θ_T")
    ax.set_ylabel("leakage  (control-margin, solid)")
    ax2.set_ylabel("linearization error  (dashed)")
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="center right", fontsize=12)
    return save(fig, "anchor_two_curve.png")


# ============================================================================ rank sweep / spectrum (Part 2)
def _cell(nc, r):
    d = torch.load(f"{RES}/jacobian_j1_ranksweep_mnist_nc{nc}_r{r}.pth", map_location="cpu", weights_only=False)
    cs = d["results"][(320, 0.01)]["colspace"]
    return {"q": {float(k): int(v) for k, v in cs["q_eff"].items()}, "sig": cs["sigma_snr"].double().numpy(),
            "rJ": int(cs["r_J"])}


def fig_rank_sweep():
    ranks = [8, 16, 32]
    b = {r: _cell(2, r) for r in ranks}
    m = {r: _cell(10, r) for r in ranks}
    assert b[8]["q"][1.0] == 59 and m[8]["q"][1.0] == 36, "anchor self-check failed (rank sweep)"
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    xs = np.log2(ranks)
    qb = [b[r]["q"][1.0] / 80 for r in ranks]
    qm = [m[r]["q"][1.0] / 80 for r in ranks]
    ax.plot(xs, qb, "-o", color=BLUE, lw=2.8, ms=10, label="binary task")
    ax.plot(xs, qm, "-s", color=ORANGE, lw=2.8, ms=10, label="10-class task")
    ax.fill_between(xs, qm, qb, color=ORANGE, alpha=0.12)
    ax.axvline(np.log2(10), color=GRAY, ls="--", lw=1.5)
    ax.text(np.log2(10) + 0.05, 0.04, "r = 10 = √(K·N): 10-class threshold\n(binary threshold ≈ √N ≈ 3; Jang 2024)", color=GRAY, fontsize=12)
    for x, y1, y2, r in zip(xs, qb, qm, ranks):
        ax.annotate(f"gap {int(round((y1-y2)*80))}", ((x), (y1 + y2) / 2), fontsize=13, color=GRAY,
                    ha="left", xytext=(8, 0), textcoords="offset points")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"r = {r}" for r in ranks])
    ax.set_ylim(0, 1)
    ax.set_xlabel("LoRA rank  (N = 10 private images, 8 directions each)")
    ax.set_ylabel("recoverable fraction of private directions")
    ax.legend(frameon=False, loc="center right")
    print(f"[rank] q_eff@ε1 binary {[b[r]['q'][1.0] for r in ranks]}  10-class {[m[r]['q'][1.0] for r in ranks]}")
    return save(fig, "rank_sweep.png")


def fig_spectrum():
    b, m = _cell(2, 8), _cell(10, 8)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    idx = np.arange(1, 81)
    ax.semilogy(idx, np.sort(b["sig"])[::-1], "o-", color=BLUE, ms=4, lw=1.8, label="binary task")
    ax.semilogy(idx, np.sort(m["sig"])[::-1], "s-", color=ORANGE, ms=4, lw=1.8, label="10-class task")
    ax.axhline(1.0, color=RED, ls="--", lw=2)
    ax.text(2, 1.18, "noise floor  (ε = 1)", color=RED, fontsize=13)
    ax.set_xlabel("private direction, sorted")
    ax.set_ylabel("signal-to-training-noise per direction")
    ax.legend(frameon=False, loc="upper right")
    return save(fig, "spectrum_r8.png")


# ============================================================================ the instrument (Part 3)
def fig_estimator():
    J = lambda p: json.load(open(os.path.join(RES, p)))
    b = J("arm_b_dilution/arm_b_summary.json")["results"]
    nd = J("arm_b_dilution/null_diag.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw=dict(width_ratios=[1, 1.5]))
    # (a) estimator drift K -> 2K (self-test, 60 synthetic datasets; whitened_sensitivity_metric.md UPDATE 2026-08-27)
    a1.bar([0, 1], [44, 6.3], color=[RED, GREEN], width=0.6)
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["2-way cross-fit\n(retracted)", "3-way cross-fit\n(used)"])
    a1.set_ylabel("drift of E[d²] when K → 2K   (%)")
    a1.axhline(15, color=GRAY, ls=":", lw=1.5)
    a1.text(1.35, 16, "gate 15%", color=GRAY, fontsize=12, ha="right")
    for x, v in ((0, 44), (1, 6.3)):
        a1.text(x, v + 1.2, f"+{v:g}%", ha="center", fontsize=15, fontweight="bold")
    a1.set_title("the estimator must converge in K")
    # (b) zero reads zero: reseed-vs-reseed vs real single-image swap
    Ns = [r["N"] for r in b]
    real = [r["whitened_sensitivity"] for r in b]
    null = {r["N"]: r["null_sensitivity"] for r in nd}
    x = np.arange(len(Ns))
    a2.bar(x - 0.2, real, width=0.4, color=BLUE, label="one image swapped  (p = 0.002 at every N)")
    a2.bar(x + 0.2, [null.get(n, np.nan) for n in Ns], width=0.4, color=GRAY, label="nothing swapped, seeds only")
    for i, n in enumerate(Ns):
        if n in null:
            a2.text(i + 0.2, 0.8, f"{null[n]:+.3f}", ha="center", fontsize=12, color=GRAY)
    a2.set_xticks(x)
    a2.set_xticklabels([f"N = {n}" for n in Ns])
    a2.set_ylabel("whitened sensitivity  d²  (debiased)")
    a2.axhline(0, color="k", lw=1)
    a2.set_ylim(0, max(real) * 1.45)
    a2.legend(frameon=False, loc="upper left", fontsize=12)
    a2.set_title("zero reads zero; one image reads loud")
    return save(fig, "estimator_honest.png")


# ============================================================================ battery (Part 4)
def fig_knobs():
    J = lambda p: json.load(open(os.path.join(RES, p)))
    b = J("arm_b_dilution/arm_b_summary.json")["results"]
    e = J("arm_e_duplication/arm_e_summary.json")
    ef = J("arm_e_duplication/arm_e_summary_fashion.json")
    d = J("arm_d_context/arm_d_summary.json")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.9))
    # B — dilution: sensitivity of ONE swapped image vs dataset size N
    Ns = [r["N"] for r in b]
    a1.plot(Ns, [r["whitened_sensitivity"] for r in b], "-o", color=BLUE, lw=2.6, ms=9)
    a1.set_xscale("log", base=2)
    a1.set_xticks(Ns)
    a1.set_xticklabels([str(n) for n in Ns])
    a1.set_xlabel("dataset size  N")
    a1.set_ylabel("sensitivity to one swapped image")
    a1.set_ylim(0, max(r["whitened_sensitivity"] for r in b) * 1.25)
    a1.set_title("B · more images around it")
    a1.text(0.04, 0.92, "detected at every N", transform=a1.transAxes, fontsize=13, color=BLUE)
    # E — duplication: copies k of the image
    for dat, key, c, lab in ((e, "8", BLUE, "MNIST"), (ef, "8", ORANGE, "Fashion")):
        rows = dat["by_rank"][key]
        ks = [r["k"] for r in rows]
        s = [r["whitened_sensitivity"] for r in rows]
        beta = dat["scaling"][key]["beta_sensitivity"]
        a2.loglog(ks, s, "o", color=c, ms=9)
        kk = np.array([1, 8.0])
        a2.loglog(kk, s[0] * kk ** beta, "-", color=c, lw=2.2, label=f"{lab}:  d² ∝ k^{beta:.2f}")
    a2.loglog([1, 8], [e["by_rank"]["8"][0]["whitened_sensitivity"] * x for x in (1, 8)], ":", color=GRAY, lw=1.5,
              label="linear  (k¹)")
    a2.set_xticks([1, 2, 4, 8])
    a2.set_xticklabels(["1", "2", "4", "8"])
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_xlabel("copies of the image  k")
    a2.set_ylabel("sensitivity")
    a2.set_title("E · duplicating it")
    a2.legend(frameon=False, fontsize=12, loc="upper left")
    # D — context rarity
    ms = [r["m"] for r in d["results"]]
    a3.plot(ms, [r["sens"] for r in d["results"]], "-o", color=BLUE, lw=2.6, ms=9)
    a3.set_xscale("log", base=2)
    a3.set_xticks(ms)
    a3.set_xticklabels([str(m) for m in ms])
    a3.set_xlabel("same-class companions  m")
    a3.set_ylabel("sensitivity of the fixed image")
    a3.set_ylim(0, max(r["sens"] for r in d["results"]) * 1.25)
    a3.set_title("D · how rare its context is  (rare → common)")
    a3.text(0.04, 0.92, f"rarity gain ≈ {d['mean_rarity_gain']:.1f}×", transform=a3.transAxes, fontsize=13, color=BLUE)
    return save(fig, "battery_knobs.png")


def fig_arm_c():
    J = lambda p: json.load(open(os.path.join(RES, p)))
    c1 = J("arm_c_imbalance/arm_c_summary.json")["results"]
    c0 = J("arm_c_imbalance/arm_c_summary_minc0.json")["results"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ms = [r["m"] for r in c1]
    ax.plot(ms, [r["sens_minority"] for r in c1], "-o", color=RED, lw=2.6, ms=9, label="class 1 image  (class 1 is the rare class)")
    ax.plot(ms, [r["sens_majority"] for r in c1], "--o", color=RED, lw=2.0, ms=7, alpha=0.6, label="class 0 image  (same run)")
    ax.plot(ms, [r["sens_majority"] for r in c0], "-s", color=BLUE, lw=2.6, ms=9, label="class 1 image  (class 0 is the rare class)")
    ax.plot(ms, [r["sens_minority"] for r in c0], "--s", color=BLUE, lw=2.0, ms=7, alpha=0.6, label="class 0 image  (same run)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("images of the rare class  m  (of N = 16)")
    ax.set_ylabel("sensitivity of one swapped image")
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    r1 = np.mean([r["sens_minority"] for r in c1]) / np.mean([r["sens_majority"] for r in c1])
    r0 = np.mean([r["sens_majority"] for r in c0]) / np.mean([r["sens_minority"] for r in c0])
    print(f"[armC] class1/class0 sensitivity ratio: class1-rare {r1:.1f}x, class0-rare {r0:.1f}x")
    return save(fig, "arm_c.png")


def fig_g0():
    s = json.load(open(f"{RES}/margin_at_scale/summary.json"))
    pt = s["per_target"]
    h = s["headline"]
    g0 = np.array([t["g0"] for t in pt])
    se = np.array([t["sensitivity"] for t in pt])
    order = np.argsort(g0)
    tier = np.zeros(len(pt), int)
    for rank, i in enumerate(order):
        tier[i] = min(2, rank * 3 // len(pt))
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    cols = [BLUE, ORANGE, RED]
    names = ["low g₀", "mid g₀", "high g₀"]
    for t in range(3):
        ax.scatter(g0[tier == t], se[tier == t], s=90, color=cols[t], edgecolor="white", lw=0.8, label=names[t], zorder=3)
    ax.set_xlabel("g₀ = ‖∇_W BCE‖ of the image at the PUBLIC base model θ₀")
    ax.set_ylabel("sensitivity of the adapter to this image")
    ax.text(0.97, 0.06, f"Spearman ρ = {h['rho_sens_g0']:+.2f}   (n = {h['n']}, 95% CI [{h['ci95'][0]:.2f}, {h['ci95'][1]:.2f}])",
            transform=ax.transAxes, ha="right", fontsize=13, color=GRAY)
    ax.legend(frameon=False, loc="upper left")
    print(f"[g0] rho {h['rho_sens_g0']:.3f} terciles {h['tercile_rhos']}  lam rho {s['mechanism_table']['rho_sens_lam']:.3f}")
    return save(fig, "g0_scatter.png")


def fig_ladder_strip():
    ts = [torch.load(f"{RES}/similarity_ladder/ladder_t{i}.pth", weights_only=False) for i in (0, 1)]
    ncol = ts[0]["T_prime_stack"].shape[0] + 1
    fig, axes = plt.subplots(2, ncol, figsize=(1.55 * ncol, 4.3))
    for r, t in enumerate(ts):
        T = t["T_img"]
        d = [float(torch.norm((t["T_prime_stack"][c] - T).flatten())) for c in range(ncol - 1)]
        sref = float(t["sensitivity"][t["rung_names"].index("r_cross")])
        order = np.argsort(d)
        axes[r, 0].imshow(T.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title("private image", fontsize=12, color=BLUE, fontweight="bold")
        for j, c in enumerate(order):
            ax = axes[r, j + 1]
            ax.imshow(t["T_prime_stack"][c].squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
            s = float(t["sensitivity"][c]) / sref
            ax.set_title(f"s = {s:.2f}", fontsize=12, color=(GRAY if s < 0.1 else "k"))
            ax.set_xlabel(f"d = {d[c]:.1f}", fontsize=10, color=GRAY)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.text(0.5, 1.01, "swap the private image for …   (left → right: more different, d = pixel distance;  s = adapter sensitivity, normalised)",
             ha="center", fontsize=13, color=GRAY)
    return save(fig, "ladder_strip.png")


def fig_h_gate():
    h = json.load(open(f"{RES}/h_spotcheck/h_spotcheck.json"))
    cells = [c for c in h["cells"] if np.isfinite(c["mem"])]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.errorbar([c["sens"] for c in cells], [c["mem"] for c in cells], yerr=[c["mem_sem"] for c in cells],
                fmt="o", ms=10, color=BLUE, ecolor=GRAY, capsize=3)
    cor = h["correlations"]
    ax.text(0.97, 0.06, f"Spearman ρ = {cor['rho_mem_sens']:+.2f}   (n = {cor['n']})", transform=ax.transAxes,
            ha="right", fontsize=13, color=GRAY)
    ax.set_xlabel("adapter sensitivity to the image  (our instrument)")
    ax.set_ylabel("behavioural memorisation\n(margin with vs without the image)")
    return save(fig, "h_gate.png")


def fig_beyond():
    J = lambda p: json.load(open(os.path.join(RES, p)))
    v = torch.load(f"{RES}/vit_lora_sensitivity/vit_lora_vit_tiny_patch16_224_r4_N16.pth", map_location="cpu",
                   weights_only=False)["metrics"]
    per = v["per_target"]
    if isinstance(per, str):
        per = eval(per)
    bf = J("arm_b_dilution/arm_b_summary_fashion.json")["results"]
    F = J("fullft_valley/F_summary.json")
    ds = J("fullft_valley/valley_headline_dstar.json")
    fig, axs = plt.subplots(1, 4, figsize=(18, 5.0))
    fig.subplots_adjust(wspace=0.42)
    a1, a2, a3, a4 = axs
    # ViT
    a1.bar(range(len(per)), [p["sensitivity"] for p in per], color=BLUE, width=0.6)
    a1.set_xticks(range(len(per)))
    a1.set_xticklabels([f"image {p['target'] + 1}" for p in per])
    a1.set_ylabel("sensitivity")
    a1.set_title(f"ViT-tiny LoRA, r={v['rank']}\n({v['n_lora_params']:,} adapter params)", fontsize=15)
    a1.text(0.5, 0.9, "each detected, p = 0.002", transform=a1.transAxes, ha="center", fontsize=13, color=BLUE)
    a1.set_ylim(0, max(p["sensitivity"] for p in per) * 1.3)
    # Fashion N sweep
    a2.bar(range(len(bf)), [r["whitened_sensitivity"] for r in bf], color=ORANGE, width=0.6)
    a2.set_xticks(range(len(bf)))
    a2.set_xticklabels([f"N={r['N']}" for r in bf])
    a2.set_title("Fashion-MNIST\none image swapped", fontsize=15)
    a2.text(0.5, 0.9, "detected at every N, p = 0.002", transform=a2.transAxes, ha="center", fontsize=13, color=ORANGE)
    a2.set_ylim(0, max(r["whitened_sensitivity"] for r in bf) * 1.3)
    # full-FT vs LoRA footprint
    pt = F["per_target"]
    full = [t["full"]["concat"]["sensitivity"] for t in pt]
    lora = [t["lora"]["concat"]["sensitivity"] for t in pt]
    a3.scatter(lora, full, s=90, color=RED, edgecolor="k", zorder=3)
    a3.set_xlabel("LoRA r=8: removal footprint")
    a3.set_ylabel("full FT: removal footprint")
    a3.set_title("full FT vs LoRA\nsame images imprint most", fontsize=15)
    a3.text(0.04, 0.9, f"rank ρ = {F['P5b_cross_regime_rank']['rho']:+.2f}", transform=a3.transAxes, fontsize=13, color=RED)
    # valley width d*
    keys = [k for k in ds["full_D_dstar"] if k in ds["lora_A_dstar"]]
    x = np.arange(len(keys))
    a4.bar(x - 0.2, [ds["lora_A_dstar"][k] for k in keys], width=0.4, color=BLUE, label="LoRA r=8")
    a4.bar(x + 0.2, [ds["full_D_dstar"][k] for k in keys], width=0.4, color=RED, label="full FT")
    a4.set_xticks(x)
    a4.set_xticklabels([str(i + 1) for i in range(len(keys))])
    a4.set_xlabel("private image")
    a4.set_ylabel("valley width  d*  (pixels)")
    a4.set_title("valley width d*\n(how far a swap must go to be seen)", fontsize=15)
    a4.legend(frameon=False, fontsize=12)
    ratios = [ds["full_D_dstar"][k] / ds["lora_A_dstar"][k] for k in keys]
    print(f"[beyond] vit sens {[round(p['sensitivity'],2) for p in per]}; d* geomean full/LoRA {np.exp(np.mean(np.log(ratios))):.2f}")
    return save(fig, "beyond_mlp.png")


def _atlas_bank(path):
    """Load the converged adapters of the atlas zoo (same filter as atlas_analyze._load)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    bank = [c for c in d["bank"] if c.get("converged", True)]
    for c in bank:
        A = c["A"].to(torch.float64).numpy()
        B = c["B"].to(torch.float64).numpy()
        c["dW"] = B @ A
        c["BA_flat"] = np.concatenate([B.ravel(), A.ravel()])
    return bank


def _atlas_distances(bank, P=8):
    """ΔW subspace distance (Grassmann U + Grassmann V + spectral cosine) and raw (B,A) cosine distance —
    numerically identical to atlas_analyze.dw_distance / ba_distance (re-implemented here only to avoid
    the scipy import that module makes at top level; SVD features computed once per adapter)."""
    import time
    t0 = time.time()
    feats = []
    for c in bank:
        U, s, Vt = np.linalg.svd(c["dW"], full_matrices=False)
        feats.append((U[:, :P], s[:P], Vt[:P].T))
    print(f"[atlas] {len(bank)} SVDs in {time.time()-t0:.1f}s", flush=True)
    n = len(feats)
    D = np.zeros((n, n))
    for i in range(n):
        Ui, si, Vi = feats[i]
        for j in range(i + 1, n):
            Uj, sj, Vj = feats[j]
            gu = np.sqrt(max(P - ((Ui.T @ Uj) ** 2).sum(), 0.0))
            gv = np.sqrt(max(P - ((Vi.T @ Vj) ** 2).sum(), 0.0))
            cos = float(np.clip((si @ sj) / (np.linalg.norm(si) * np.linalg.norm(sj) + 1e-12), -1, 1))
            D[i, j] = D[j, i] = max(0.0, gu + gv + (1 - cos))
    X = np.stack([c["BA_flat"] for c in bank])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Dba = np.sqrt(np.maximum(2 - 2 * (X @ X.T), 0))
    print(f"[atlas] distances in {time.time()-t0:.1f}s", flush=True)
    return D, Dba


def _mds(D):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 1e-12))


def fig_atlas():
    bank = _atlas_bank(os.path.join(RES, "atlas_zoo", "zoo_bank.pth"))
    Ddw, Dba = _atlas_distances(bank)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, D, key, name in ((a1, Ddw, "composition", "ΔW = BA   coloured by TRAINING DATA (5 digit-subsets)"),
                             (a2, Dba, "init_seed", "raw (B, A)   coloured by INIT SEED")):
        emb = _mds(D)
        vals = [str(c[key]) for c in bank]
        uniq = sorted(set(vals))
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 2)))
        for u, cc in zip(uniq, cmap):
            idx = [i for i, vv in enumerate(vals) if vv == u]
            ax.scatter(emb[idx, 0], emb[idx, 1], color=cc, s=46, alpha=0.85, edgecolor="k", linewidth=0.3)
        ax.set_title(name, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
    print(f"[atlas] {len(bank)} adapters")
    return save(fig, "atlas_2panel.png")


def fig_gallery():
    sets = [("mnist", "MNIST"), ("cifar10", "CIFAR-10"), ("flowers32", "Flowers")]
    fig, axes = plt.subplots(2, 6, figsize=(12.5, 4.5))
    col = 0
    for ds, lab in sets:
        d = torch.load(f"{RES}/gb_e2e_{ds}_N2_gelu.pth", map_location="cpu", weights_only=False)
        key = "TRUE ΔW (ceiling)"
        rec = d["recons"][key]
        for i in range(2):
            for row, img in ((0, d["x_cen"][i]), (1, rec[i])):
                x = np.clip((img + d["ds_mean"][0]).detach().cpu().numpy(), 0, 1)
                ax = axes[row, col]
                ax.imshow(x[0] if x.shape[0] == 1 else np.transpose(x, (1, 2, 0)), cmap="gray" if x.shape[0] == 1 else None,
                          vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if i == 0:
                axes[0, col].set_title(lab, fontsize=15, loc="left", color=BLUE, fontweight="bold")
            col += 1
    axes[0, 0].set_ylabel("private image", fontsize=14)
    axes[1, 0].set_ylabel("reconstructed", fontsize=14)
    return save(fig, "gallery.png")


FIGS = {
    "crux_bars": fig_crux_bars, "fs_vs_T": fig_fs_vs_T, "anchor": fig_anchor, "rank_sweep": fig_rank_sweep,
    "spectrum": fig_spectrum, "estimator": fig_estimator, "knobs": fig_knobs, "arm_c": fig_arm_c, "g0": fig_g0,
    "ladder": fig_ladder_strip, "h_gate": fig_h_gate, "beyond": fig_beyond, "atlas": fig_atlas, "gallery": fig_gallery,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated subset of: " + ",".join(FIGS))
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(FIGS)
    for n in names:
        FIGS[n]()
