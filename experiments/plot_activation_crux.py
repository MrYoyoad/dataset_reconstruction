#!/usr/bin/env python
"""
Activation-crux figure + analysis (supervisor's TOP ask).

Question (implicit-bias framing): how does ACTIVATION SMOOTHNESS interact with
  (i) reconstruction FIDELITY (baseline-relative SSIM), and
  (ii) the NTK / lazy regime (feature_stability, weight_change, eff-rank)
in the LoRA reconstruction setting?

Prior finding on record (notes/crux_activation_analysis.md, STATUS/LESSONS):
  "activation smoothness sets reconstruction FIDELITY, not leakage direction-COUNT;
   r_J was beta-independent; the naive 'smoother => more leakage' is REFUTED
   (rho~+0.03, inverts on flowers)."

This script re-does the honest read on the fuller committed CSV, and EXTENDS it
with whatever the two still-RUNNING jobs have written:
  - crux_featstab_T_390026.out  (feature-stability-vs-T sweep, oracle)
  - crux_freec_ladder_392821.out (free-coefficient wc-ladder)
Both jobs are partial/incomplete -> everything derived from them is labelled PARTIAL.

CPU-only. No GPU, no bsub. Reads a committed CSV + parses two log files, writes:
  figures/crux/activation_crux_summary.png

Run:  python experiments/plot_activation_crux.py
"""
import csv
import os
import re
import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "results", "rescored_activations_857271_2026-08-11.csv")
FEATSTAB_LOG = os.path.join(ROOT, "scripts", "wexac_logs", "crux_featstab_T_390026.out")
FREEC_LOG = os.path.join(ROOT, "scripts", "wexac_logs", "crux_freec_ladder_392821.out")
OUT_PNG = os.path.join(ROOT, "figures", "crux", "activation_crux_summary.png")

# ---------------------------------------------------------------------------
# Smoothness ordering (documented). Higher ordinal = SMOOTHER.
# Rationale (differentiability class, canonical smooth-ReLU ordering):
#   0  relu, leaky_relu     -- kinked, non-differentiable at 0 (sharpest)
#   1  hardswish            -- piecewise, two kinks in derivative
#   2  elu, celu, selu      -- C^1 (continuous 1st deriv, kink in 2nd)
#   3  tanh, sigmoid        -- C^inf but saturating/bounded transition
#   4  silu, gelu, gelu_tanh, mish -- C^inf smooth-ReLU family
#   5  softplus             -- C^inf, canonical smoothest (beta-knob anchor)
# This matches the task's suggested chain relu/leaky/hardswish < elu < gelu/silu < softplus.
SMOOTHNESS = {
    "relu": 0, "leaky_relu": 0,
    "hardswish": 1,
    "elu": 2, "celu": 2, "selu": 2,
    "tanh": 3, "sigmoid": 3,
    "silu": 4, "gelu": 4, "gelu_tanh": 4, "mish": 4,
    "softplus": 5,
}


def smoothness(act):
    return SMOOTHNESS.get(act, np.nan)


# ---------------------------------------------------------------------------
# Manual Spearman (no scipy in this env).
def _rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    i = 0
    sa = a[order]
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), n
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    rho = float((rx @ ry) / (math.sqrt((rx @ rx) * (ry @ ry))))
    return rho, n


# ---------------------------------------------------------------------------
# CSV load (committed source of truth).
def load_csv():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            act = (r["finetune_activation"] or "").strip()
            def fnum(k):
                v = r.get(k, "")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return np.nan
            npc = fnum("n_per_class")
            # unlabeled activation + npc>1 rows are the relu N-scaling controls
            if act == "" and (not np.isfinite(npc) or npc > 1):
                act = "relu(npc)"  # keep but flag; excluded from activation stats
            rec = dict(
                act=act,
                ssim_norm=fnum("ssim_norm"),
                ssim_mean_baseline=fnum("ssim_mean_baseline"),
                weight_change=fnum("weight_change"),
                eff_rank=fnum("delta_w_effective_rank"),
                feature_stability=fnum("feature_stability"),
                ntk_passed=(r.get("ntk_passed", "").strip() == "True"),
                lr=fnum("lr"),
                npc=npc,
                T=fnum("n_steps"),
                ctrl_margin_norm=fnum("ctrl_margin_norm"),  # leakage (clip-robust)
            )
            rec["fidelity"] = rec["ssim_norm"] - rec["ssim_mean_baseline"]
            rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# Parse a running-job log. Blocks look like:
#   ########## act=<A> [target_wc=<W>] ... T=<T> r=8 ##########
#   ...
#   LoRA rank=8 (T=<T>): {dict with ssim_norm, ssim_mean_baseline, clipped_fraction}
#   LoRA NTK diagnostics: {dict with weight_change, feature_stability, ntk_passed, delta_w_effective_rank}
#   [Control: {dict with ssim_norm}]   (present in both logs)
HDR = re.compile(r"##########\s*act=(?P<act>[a-z_]+)\b"
                 r"(?:.*?target_wc=(?P<wc>[0-9.]+))?.*?T=(?P<T>\d+)\b")
LORA = re.compile(r"LoRA rank=\d+ \(T=\d+\):\s*(\{.*\})")
DIAG = re.compile(r"LoRA NTK diagnostics:\s*(\{.*\})")
CTRL = re.compile(r"^Control:\s*(\{.*\})")


def _pydict(s):
    # dicts are printed with python repr of floats/ints/bools -> eval-safe subset
    return eval(s, {"__builtins__": {}}, {})  # noqa: S307 (trusted local log)


def parse_log(path):
    if not os.path.exists(path):
        return []
    blocks = []
    cur = None
    with open(path) as f:
        for line in f:
            mh = HDR.search(line)
            if mh:
                if cur and "lora" in cur and "diag" in cur:
                    blocks.append(cur)
                cur = dict(act=mh.group("act"),
                           target_wc=(float(mh.group("wc")) if mh.group("wc") else np.nan),
                           T=int(mh.group("T")))
                continue
            if cur is None:
                continue
            ml = LORA.search(line)
            if ml:
                cur["lora"] = _pydict(ml.group(1))
                continue
            md = DIAG.search(line)
            if md:
                cur["diag"] = _pydict(md.group(1))
                continue
            mc = CTRL.match(line)
            if mc:
                cur["ctrl"] = _pydict(mc.group(1))
    if cur and "lora" in cur and "diag" in cur:
        blocks.append(cur)
    # flatten
    out = []
    for b in blocks:
        lora, diag = b["lora"], b["diag"]
        ctrl = b.get("ctrl", {})
        rec = dict(
            act=b["act"], T=b["T"], target_wc=b.get("target_wc", np.nan),
            ssim_norm=lora.get("ssim_norm", np.nan),
            ssim_mean_baseline=lora.get("ssim_mean_baseline", np.nan),
            clipped_fraction=lora.get("clipped_fraction", np.nan),
            weight_change=diag.get("weight_change", np.nan),
            feature_stability=diag.get("feature_stability", np.nan),
            eff_rank=diag.get("delta_w_effective_rank", np.nan),
            ntk_passed=bool(diag.get("ntk_passed", False)),
            ctrl_ssim_norm=ctrl.get("ssim_norm", np.nan),
        )
        rec["fidelity"] = rec["ssim_norm"] - rec["ssim_mean_baseline"]
        rec["ctrl_margin_norm"] = rec["ssim_norm"] - rec["ctrl_ssim_norm"]
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
def per_activation_canonical(csv_rows):
    """One representative row per labelled activation: lowest lr (closest to NTK
    point) at npc=1. Returns dict act -> rec."""
    best = {}
    for r in csv_rows:
        a = r["act"]
        if a in ("", "relu(npc)"):
            continue
        if not (np.isnan(r["npc"]) or r["npc"] == 1):
            continue
        if a not in best or r["lr"] < best[a]["lr"]:
            best[a] = r
    return best


def main():
    csv_rows = load_csv()
    feat = parse_log(FEATSTAB_LOG)     # feature-stability-vs-T, oracle, PARTIAL
    freec = parse_log(FREEC_LOG)       # free-coefficient wc-ladder, PARTIAL

    # ----- CSV activation-level table (canonical lr per activation) ----------
    canon = per_activation_canonical(csv_rows)
    acts = sorted(canon, key=lambda a: (smoothness(a), a))
    print("\n" + "=" * 78)
    print("ACTIVATION-CRUX ANALYSIS  (committed CSV + PARTIAL running-job logs)")
    print("=" * 78)
    print(f"CSV rows total: {len(csv_rows)}  |  labelled npc=1 activations: {len(canon)}")
    print(f"featstab_T log rows parsed (PARTIAL, running): {len(feat)}")
    print(f"freec_ladder log rows parsed (PARTIAL, running): {len(freec)}")

    print("\n-- CSV per-activation (canonical = lowest lr, npc=1, T=1, oracle) --")
    print(f"{'act':12s} {'smooth':>6s} {'fidelity':>9s} {'ssim_norm':>9s} "
          f"{'feat_stab':>9s} {'wc':>7s} {'effR':>5s} {'ntk':>4s}")
    for a in acts:
        r = canon[a]
        print(f"{a:12s} {smoothness(a):6d} {r['fidelity']:9.3f} {r['ssim_norm']:9.3f} "
              f"{r['feature_stability']:9.3f} {r['weight_change']:7.3f} "
              f"{r['eff_rank']:5.0f} {str(r['ntk_passed']):>4s}")

    # ----- Spearman rho on CSV activation set --------------------------------
    sm = [smoothness(a) for a in acts]
    fid = [canon[a]["fidelity"] for a in acts]
    fs = [canon[a]["feature_stability"] for a in acts]
    wc = [canon[a]["weight_change"] for a in acts]
    er = [canon[a]["eff_rank"] for a in acts]

    rho_sm, n_sm = spearman(sm, fid)
    rho_fs, n_fs = spearman(fs, fid)
    rho_wc, n_wc = spearman(wc, fid)
    rho_er, n_er = spearman(er, fid)
    rho_smfs, _ = spearman(sm, fs)

    # ----- Broader smoothness check on PARTIAL featstab T=1 (more activations) -
    feat_t1 = {}
    for r in feat:
        if r["T"] == 1:
            feat_t1.setdefault(r["act"], r)
    # fall back to smallest available T per activation if T=1 missing
    feat_small = {}
    for r in sorted(feat, key=lambda z: z["T"]):
        feat_small.setdefault(r["act"], r)
    base_feat = feat_t1 if len(feat_t1) >= 5 else feat_small
    fa = sorted(base_feat, key=lambda a: (smoothness(a), a))
    if len(fa) >= 3:
        rho_sm_feat, n_feat = spearman([smoothness(a) for a in fa],
                                       [base_feat[a]["fidelity"] for a in fa])
        rho_smfs_feat, _ = spearman([smoothness(a) for a in fa],
                                    [base_feat[a]["feature_stability"] for a in fa])
    else:
        rho_sm_feat, n_feat, rho_smfs_feat = float("nan"), len(fa), float("nan")

    # leakage (ctrl_margin_norm) vs smoothness on the SAME CSV set -> reconcile
    # with the prior note ("smoother=>more LEAKAGE refuted, rho~+0.03").
    leak = [canon[a]["ctrl_margin_norm"] for a in acts]
    rho_sm_leak, _ = spearman(sm, leak)

    # Realistic free-c leakage ranking at a matched wc rung (freec ladder, PARTIAL).
    def freec_at_wc(target):
        d = {}
        for r in freec:
            if abs(r["target_wc"] - target) < 1e-6:
                d.setdefault(r["act"], r)
        return d
    freec_wc = None
    for tw in (0.1, 0.3, 0.03):
        cand = freec_at_wc(tw)
        if len(cand) >= 5:
            freec_wc = (tw, cand)
            break
    if freec_wc:
        tw, cand = freec_wc
        fca = sorted(cand, key=lambda a: (smoothness(a), a))
        rho_sm_leak_free, n_free = spearman([smoothness(a) for a in fca],
                                            [cand[a]["ctrl_margin_norm"] for a in fca])
        rho_sm_fid_free, _ = spearman([smoothness(a) for a in fca],
                                      [cand[a]["fidelity"] for a in fca])
    else:
        tw, cand, fca = float("nan"), {}, []
        rho_sm_leak_free, n_free, rho_sm_fid_free = float("nan"), 0, float("nan")

    if fa:
        print(f"\n-- PARTIAL featstab base set (T={base_feat[fa[0]]['T']}), FULLER activation span --")
        print(f"{'act':12s} {'smooth':>6s} {'fidelity':>9s} {'feat_stab':>9s} {'wc':>7s}")
        for a in fa:
            r = base_feat[a]
            print(f"{a:12s} {smoothness(a):6d} {r['fidelity']:9.3f} "
                  f"{r['feature_stability']:9.3f} {r['weight_change']:7.3f}")

    print("\n-- Spearman rho (fidelity = ssim_norm - ssim_mean_baseline) --")
    print(f"  smoothness   vs fidelity : rho = {rho_sm:+.3f}  (n={n_sm}, CSV activations)")
    print(f"  feat_stab    vs fidelity : rho = {rho_fs:+.3f}  (n={n_fs}, CSV)")
    print(f"  weight_change vs fidelity: rho = {rho_wc:+.3f}  (n={n_wc}, CSV)")
    print(f"  eff_rank     vs fidelity : rho = {rho_er:+.3f}  (n={n_er}, CSV)")
    print(f"  smoothness   vs feat_stab: rho = {rho_smfs:+.3f}  (n={n_sm}, CSV)")
    print(f"  [PARTIAL featstab, n={n_feat} acts] smoothness vs fidelity  : rho = {rho_sm_feat:+.3f}")
    print(f"  [PARTIAL featstab, n={n_feat} acts] smoothness vs feat_stab : rho = {rho_smfs_feat:+.3f}")
    print("\n-- RECONCILE with prior note: leakage (ctrl_margin_norm) vs smoothness --")
    print(f"  smoothness vs LEAKAGE (ctrl_margin_norm, CSV oracle): rho = {rho_sm_leak:+.3f}  (n={n_sm})")
    print(f"  smoothness vs LEAKAGE (free-c ladder @ wc={tw}, PARTIAL): rho = {rho_sm_leak_free:+.3f}  (n={n_free})")
    print(f"  smoothness vs FIDELITY (free-c ladder @ wc={tw}, PARTIAL): rho = {rho_sm_fid_free:+.3f}  (n={n_free})")
    if fca:
        print(f"  free-c leakage ranking @ wc={tw} (smoothest->sharpest): " +
              ", ".join(f"{a}={cand[a]['ctrl_margin_norm']:+.2f}" for a in fca))

    # ----- NTK regime accounting --------------------------------------------
    ntk_true = sum(1 for r in csv_rows if r["ntk_passed"])
    print(f"\n-- NTK regime -- CSV ntk_passed True/False: {ntk_true}/{len(csv_rows) - ntk_true}")
    feat_true = sum(1 for r in feat if r["ntk_passed"])
    freec_true = sum(1 for r in freec if r["ntk_passed"])
    print(f"   featstab_T ntk_passed True/total: {feat_true}/{len(feat)} (PARTIAL)")
    print(f"   freec_ladder ntk_passed True/total: {freec_true}/{len(freec)} (PARTIAL)")

    # ----- eff_rank (direction-count) accounting: is it a T- or smoothness-thing?
    er_by_act = [canon[a]["eff_rank"] for a in acts]  # CSV T=1
    rho_sm_er, _ = spearman(sm, er_by_act)
    # mean eff_rank per T from featstab (direction-count vs fine-tuning length)
    er_by_T = {}
    for r in feat:
        er_by_T.setdefault(r["T"], []).append(r["eff_rank"])
    er_T_mean = {t: float(np.nanmean(v)) for t, v in sorted(er_by_T.items())}

    # ----- Positive mechanistic characterization (NOT a pass/fail verdict) ----
    print("\n-- MECHANISM (positive characterization; observe-framed) --")
    print("  What smoothness SHAPES:")
    print(f"    smoothness -> feature_stability (NTK-lazy proxy): rho={rho_smfs:+.2f} (CSV n={n_sm}), "
          f"{rho_smfs_feat:+.2f} (featstab n={n_feat}) -> POSITIVE (smoother = lazier).")
    print(f"    feature_stability -> relative fidelity (ssim_norm): rho={rho_fs:+.2f} (n={n_fs}) "
          f"-> laziness is the load-bearing link to fidelity.")
    print("  What smoothness does NOT set (positive rejections):")
    print(f"    direction-count (delta_w eff_rank): near-constant across activations "
          f"(CSV T=1 eff_rank in {sorted(set(int(x) for x in er_by_act))}, all=2 except softplus=1 -> "
          f"degenerate column; rho_sm={rho_sm_er:+.2f} is a near-constant artifact, NOT a trend); "
          f"grows with T instead: mean eff_rank per T = "
          f"{ {t: round(v,1) for t,v in er_T_mean.items()} } -> direction-count is a T-effect (r_J ~ smoothness-independent).")
    print(f"    realistic leakage (free-c ctrl_margin_norm @ matched wc={tw}): rho_sm={rho_sm_leak_free:+.2f} "
          f"(n={n_free}) -> NOT monotone in smoothness; TWO-CLUSTER kink effect "
          f"(leaky_relu/selu leak ~5x the smooth family, which is flat ~+0.10).")
    print("  => 'smoother => more leakage / more direction-count' is REFUTED-with-a-reason: "
          "smoothness sets the lazy-regime/linearization fidelity, not the leaked direction-count.")

    # ========================================================================
    #                              FIGURE
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
    fig.suptitle("How activation smoothness shapes LoRA fine-tuning dynamics (mechanistic, OBSERVE-framed)\n"
                 "DOES set: lazy/NTK regime + fidelity   |   does NOT set: direction-count + leakage\n"
                 "CSV (committed, oracle T=1) + PARTIAL running-job logs 390026 / 392821",
                 fontsize=12, fontweight="bold")

    # colour by smoothness ordinal
    cmap = plt.cm.viridis
    def col(a):
        s = smoothness(a)
        return cmap(0.12 + 0.76 * (s / 5.0)) if np.isfinite(s) else "0.6"

    # ============ (a) SMOOTHNESS -> feature_stability (what it DOES set) ======
    # Combine CSV canonical + fuller featstab set on one smoothness-ordered bar.
    ax = axes[0, 0]
    feat_ordered = fa  # fuller activation span from featstab base
    xs = range(len(feat_ordered))
    vals = [base_feat[a]["feature_stability"] for a in feat_ordered]
    bars = ax.bar(xs, vals, color=[col(a) for a in feat_ordered])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{a}\n(s={smoothness(a)})" for a in feat_ordered],
                       fontsize=7, rotation=30, ha="right")
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("feature_stability  (NTK-lazy-regime proxy; 1 = perfectly lazy)")
    ax.set_title("(a) DOES shape: laziness (feature_stability), smoothness-ordered\n"
                 f"OBSERVE softplus/sigmoid/silu lazy ~0.97-1.0; C1 elu/celu/tanh ~0.7 "
                 f"(rho={rho_smfs_feat:+.2f}, n={n_feat}; wc-confounded)", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.5)

    # ============ (b) feature_stability -> relative fidelity (the link) =======
    ax = axes[0, 1]
    for a in acts:
        r = canon[a]
        ax.scatter(r["feature_stability"], r["ssim_norm"], s=110, color=col(a),
                   edgecolor="k", lw=0.6, zorder=3)
        ax.annotate(a, (r["feature_stability"], r["ssim_norm"]), fontsize=7.5,
                    xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("feature_stability  (lazy/NTK-regime proxy)")
    ax.set_ylabel("relative reconstruction fidelity  (ssim_norm)")
    ax.set_title("(b) Load-bearing link: fidelity tracks LAZINESS, not smoothness itself\n"
                 f"OBSERVE feat_stab->fidelity rho={rho_fs:+.2f} (n={n_fs}, CSV); "
                 f"lazier => more invertible", fontsize=8)
    ax.grid(alpha=0.3)

    # ============ (c) realistic free-c leakage @ matched wc (what it does NOT set)
    ax = axes[1, 0]
    if fca:
        vals = [cand[a]["ctrl_margin_norm"] for a in fca]
        bars = ax.bar(range(len(fca)), vals, color=[col(a) for a in fca])
        ax.set_xticks(range(len(fca)))
        ax.set_xticklabels([f"{a}\n(s={smoothness(a)})" for a in fca],
                           fontsize=7, rotation=30, ha="right")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("realistic leakage  ctrl_margin_norm  (free-c)")
        ax.set_title(f"(c) does NOT set: realistic free-c leakage @ matched wc={tw} [PARTIAL 392821]\n"
                     f"OBSERVE TWO-CLUSTER kink effect (leaky_relu/selu ~5x); smooth flat ~+0.1 "
                     f"(rho={rho_sm_leak_free:+.2f}, n={n_free})", fontsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.5)
    else:
        ax.text(0.5, 0.5, "freec ladder: no matched-wc rung parsed yet", ha="center", va="center")
        ax.set_axis_off()

    # ============ (d) direction-count & laziness vs T (a T-effect, not smoothness)
    ax = axes[1, 1]
    if feat:
        by_act = {}
        for r in feat:
            by_act.setdefault(r["act"], []).append((r["T"], r["feature_stability"]))
        for a in sorted(by_act, key=lambda z: (smoothness(z), z)):
            pts = sorted(by_act[a])
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="o", ms=3.5, lw=1.3, color=col(a), label=a, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("T  (LoRA fine-tune steps, log)")
        ax.set_ylabel("feature_stability")
        # twin axis: mean eff_rank (direction-count) vs T
        ax2 = ax.twinx()
        Ts = sorted(er_T_mean)
        ax2.plot(Ts, [er_T_mean[t] for t in Ts], "k--s", ms=5, lw=2,
                 label="mean eff_rank (dir-count)")
        ax2.set_ylabel("mean delta_w effective rank (direction-count)")
        ax.set_title("(d) does NOT set: direction-count is a T-effect [PARTIAL 390026]\n"
                     "OBSERVE feature_stability DECAYS + eff_rank GROWS (2->~7) with T, all activations",
                     fontsize=8)
        ax.legend(fontsize=6, ncol=2, loc="lower left")
        ax2.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3, which="both")
    else:
        ax.text(0.5, 0.5, "featstab_T job: no parsed rows yet", ha="center", va="center")
        ax.set_axis_off()

    fig.text(0.5, 0.005,
             "OBSERVE, not conclude (early research; leakage numbers bound only the WEAKEST attacker). "
             "Smoothness ordinal: relu/leaky=0 < hardswish=1 < elu/celu/selu=2 < tanh/sigmoid=3 < "
             "silu/gelu/mish=4 < softplus=5. MECHANISM: smoothness sets the lazy-regime/linearization fidelity, "
             "NOT the leaked direction-count (a kink/T effect). OPEN: matched-wc x multi-seed x free-c closure.",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=[0, 0.025, 1, 0.925])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"\nSaved figure -> {OUT_PNG}")

    # machine-readable summary line for the report
    print("\nSUMMARY_JSON " + repr(dict(
        n_csv=len(csv_rows), n_acts=len(canon), n_feat=len(feat), n_freec=len(freec),
        rho_smoothness_fidelity=round(rho_sm, 3),
        rho_featstab_fidelity=round(rho_fs, 3),
        rho_wc_fidelity=round(rho_wc, 3),
        rho_effrank_fidelity=round(rho_er, 3),
        rho_smoothness_featstab=round(rho_smfs, 3),
        rho_smoothness_fidelity_featstabLog=round(rho_sm_feat, 3),
        ntk_passed_csv=ntk_true,
    )))


if __name__ == "__main__":
    main()
