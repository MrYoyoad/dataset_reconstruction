#!/usr/bin/env python
"""
FIGURE 4 — the noise-free Jacobian: J_full vs J_LoRA (arm G).

Parses the arm-G stdout (there is no JSON) and renders, for the SAME tangent
basis / same D / same theta0 in both regimes:
  (a) the singular-value RANGE [sigma_max, sigma_min] of J_full vs J_LoRA across
      the mandatory T-sweep (log-y), with r_J and eff_rank annotated. (Only the
      spectrum ENDPOINTS are logged, so we plot the range, not every sigma_i.)
  (b) the P7 valley ratio ||J . a_nn|| / ||J . a_far|| — the local-linear analogue
      of s(d) — as grouped bars (full vs LoRA) across the T-sweep.
  (c) per-layer col(J_full) energy of the top singular vector across L0/L1/L2.

This is the SECOND, independent method (local-linear Jacobian) that meets the
finite-swap dial (Figs 1-2): both say full-FT valley width ~ LoRA valley width.

DATA (committed, read-only; CPU render, no experiment):
  scripts/wexac_logs/fullft_jacobian_501620.out  (arm G [G] lines; all-layer
      config layers=(0,1,2) is the headline, single-layer (0,) shown as context).

FRAMING: OBSERVE, do not conclude; weakest-attacker scoped. Every arm-G readout
carries the "early-training Jacobian, NOT the converged valley" caveat (max_bce
per T is well above convergence).
"""
import os
import re
import argparse

LOG = "/home/projects/galvardi/yoado/scripts/wexac_logs/fullft_jacobian_501620.out"
FIGURES = "/home/projects/galvardi/yoado/figures/fullft_valley"

_HDR = re.compile(r"\[G\]\s*T=(\d+)\s+N=\d+\s+k=\d+\s+Nk=\d+\s+layers=\(([^)]*)\)")
_JLINE = re.compile(
    r"\[G\]\s+(J_full|J_LoRA)\s+shape=\([^)]*\)\s+r_J=(\d+)/(\d+)\s+eff_rank=([\d.]+)"
    r".*?\[([0-9.eE+-]+),\s*([0-9.eE+-]+)\]\s+max_bce=([\d.eE+-]+)")
_P7 = re.compile(
    r"\[G\]\s+P7.*full=([\d.]+)\s+LoRA=([\d.]+)")
_ENERGY = re.compile(
    r"per-layer col\(J_full\) energy.*L0=([\d.]+)\s+L1=([\d.]+)\s+L2=([\d.]+)")


def parse_log(path):
    """Return list of blocks: dict(T, layers, full=..., lora=..., p7=(f,l), energy=...)."""
    blocks = []
    cur = None
    with open(path, errors="replace") as f:
        for line in f:
            m = _HDR.search(line)
            if m:
                if cur:
                    blocks.append(cur)
                layers = tuple(int(x) for x in re.findall(r"\d+", m.group(2)))
                cur = {"T": int(m.group(1)), "layers": layers,
                       "full": None, "lora": None, "p7": None, "energy": None}
                continue
            if cur is None:
                continue
            mj = _JLINE.search(line)
            if mj:
                # sigma printed as [max, min]
                rec = {"r_J": int(mj.group(2)), "r_J_max": int(mj.group(3)),
                       "eff_rank": float(mj.group(4)),
                       "sigma_max": float(mj.group(5)),
                       "sigma_min": float(mj.group(6)),
                       "max_bce": float(mj.group(7))}
                cur["full" if mj.group(1) == "J_full" else "lora"] = rec
                continue
            mp = _P7.search(line)
            if mp:
                cur["p7"] = (float(mp.group(1)), float(mp.group(2)))
                continue
            me = _ENERGY.search(line)
            if me:
                cur["energy"] = (float(me.group(1)), float(me.group(2)),
                                 float(me.group(3)))
    if cur:
        blocks.append(cur)
    return blocks


def main():
    ap = argparse.ArgumentParser(description="Build arm-G Jacobian spectra figure.")
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--layers", default="0,1,2",
                    help="headline layer config to plot (comma list).")
    ap.add_argument("--out", default=os.path.join(FIGURES, "fig_jacobian_spectra.png"))
    args = ap.parse_args()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "grid.linewidth": 0.5})

    blocks = parse_log(args.log)
    if not blocks:
        raise RuntimeError(f"No [G] blocks parsed from {args.log}")
    want = tuple(int(x) for x in args.layers.split(","))
    hl = sorted([b for b in blocks if b["layers"] == want], key=lambda b: b["T"])
    if not hl:
        raise RuntimeError(f"No blocks for layers={want}; have "
                           f"{sorted({b['layers'] for b in blocks})}")
    print(f"[fig_jacobian_spectra] parsed {len(blocks)} blocks; headline "
          f"layers={want} T-sweep={[b['T'] for b in hl]}", flush=True)

    Ts = [b["T"] for b in hl]
    x = np.arange(len(Ts))
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(16, 5.2),
                                        gridspec_kw={"width_ratios": [1.2, 1.0, 0.85]})

    # ---- (a) singular-value range bars, full vs LoRA -----------------------
    w = 0.34
    for off, key, col, lab in [(-w / 2, "full", "#d62728", "J_full"),
                               (w / 2, "lora", "#1f77b4", "J_LoRA")]:
        lo = [b[key]["sigma_min"] for b in hl]
        hi = [b[key]["sigma_max"] for b in hl]
        # a "range bar" from sigma_min to sigma_max on log-y
        axa.bar(x + off, np.array(hi) - np.array(lo), width=w, bottom=lo,
                color=col, alpha=0.35, edgecolor=col, label=lab)
        axa.plot(x + off, hi, "_", color=col, ms=14, mew=2)
        axa.plot(x + off, lo, "_", color=col, ms=14, mew=2)
        for xi, b in zip(x, hl):
            axa.annotate(f"eff={b[key]['eff_rank']:.1f}\n$r_J$={b[key]['r_J']}",
                         (xi + off, b[key]["sigma_max"]),
                         textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=6, color=col)
    axa.set_yscale("log")
    axa.set_xticks(x)
    axa.set_xticklabels([f"T={t}" for t in Ts])
    axa.set_ylabel(r"singular value $\sigma$  (range $[\sigma_{\min},\sigma_{\max}]$)")
    axa.set_title(f"(a) J spectra: full vs LoRA, layers={want}", fontsize=10,
                  fontweight="bold")
    axa.legend(fontsize=8, loc="upper left")

    # ---- (b) P7 valley ratio bars ------------------------------------------
    pf = [b["p7"][0] for b in hl]
    pl = [b["p7"][1] for b in hl]
    axb.bar(x - w / 2, pf, width=w, color="#d62728", label="full")
    axb.bar(x + w / 2, pl, width=w, color="#1f77b4", label="LoRA")
    for xi, v in zip(x - w / 2, pf):
        axb.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w / 2, pl):
        axb.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    axb.axhline(1.0, color="gray", lw=0.8, ls=":")
    axb.set_xticks(x)
    axb.set_xticklabels([f"T={t}" for t in Ts])
    axb.set_ylim(0.9, max(pf + pl) + 0.08)
    axb.set_ylabel(r"$\|J\cdot a_{nn}\| / \|J\cdot a_{far}\|$  (P7 valley ratio)")
    axb.set_title("(b) P7 valley ratio: full $\\approx$ LoRA", fontsize=10,
                  fontweight="bold")
    axb.legend(fontsize=8, loc="upper left")

    # ---- (c) per-layer col(J_full) energy stack ----------------------------
    have_e = [b for b in hl if b["energy"] is not None]
    if have_e:
        xe = np.arange(len(have_e))
        L0 = np.array([b["energy"][0] for b in have_e])
        L1 = np.array([b["energy"][1] for b in have_e])
        L2 = np.array([b["energy"][2] for b in have_e])
        axc.bar(xe, L0, color="#1f77b4", label="L0")
        axc.bar(xe, L1, bottom=L0, color="#ff7f0e", label="L1")
        axc.bar(xe, L2, bottom=L0 + L1, color="#2ca02c", label="L2")
        for xi, a, b_, c in zip(xe, L0, L1, L2):
            axc.text(xi, a / 2, f"{a:.2f}", ha="center", va="center", fontsize=7, color="w")
        axc.set_xticks(xe)
        axc.set_xticklabels([f"T={b['T']}" for b in have_e])
        axc.set_ylabel("energy fraction of top left-singular vector")
        axc.set_title("(c) col(J_full) energy by layer", fontsize=10, fontweight="bold")
        axc.legend(fontsize=8, loc="upper right")
    else:
        axc.axis("off")

    fig.suptitle("Two independent methods converge: the noise-free Jacobian says "
                 "full-FT valley $\\approx$ LoRA valley", fontsize=12.5, fontweight="bold")

    # caption uses the T=20 all-layer numbers if present
    b20 = next((b for b in hl if b["T"] == 20), hl[-1])
    b5 = next((b for b in hl if b["T"] == 5), None)
    p7_txt = f"T=20 full={b20['p7'][0]:.3f} vs LoRA={b20['p7'][1]:.3f}"
    if b5:
        p7_txt += f"; T=5 full={b5['p7'][0]:.3f} vs LoRA={b5['p7'][1]:.3f}"
    cap = (
        "What we OBSERVE: the raw (noise-free) singular spectra of J_full and J_LoRA overlap in "
        "range and effective rank, and the P7 valley ratio ||J.a_nn||/||J.a_far|| is ~equal in "
        f"both regimes ({p7_txt}). So the local-linear Jacobian agrees with the finite-swap dial "
        "(Figs 1-2): full training does not resolve individual images into a tighter valley than "
        "a rank-8 adapter. col(J_full) energy is concentrated in L0 (panel c), echoing the depth "
        "read (Fig 2). CAVEAT: early-training Jacobian (T<=20, max_bce well above convergence) — "
        "NOT the converged T=1000 valley. WEAKEST-ATTACKER footer: bounds only the prior-free "
        "adapter-only per-image attacker; a lower bound on leakage, not the reconstruction limit."
    )
    fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=7.7, wrap=True)

    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_jacobian_spectra] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
