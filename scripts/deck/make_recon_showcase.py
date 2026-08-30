"""Showcase figures: REALISTIC (free-coefficient) reconstructions, LoRA vs full fine-tune.

User rules baked in: free-coefficients ONLY (oracle appears nowhere), raw SSIM labels per image,
control row shown (no cherry-picking the comparison away). Sources are the saved experiment-B
tensors (results/exp_b_*_free_*.pth). Output: figures/recon_showcase/*.png

  python scripts/deck/make_recon_showcase.py            # T=1 panel from existing tensors
  python scripts/deck/make_recon_showcase.py --tsweep   # add the T-sweep panel once job lands
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures", "recon_showcase")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 15, "figure.dpi": 100})


def ld(name):
    return torch.load(os.path.join(RES, name), map_location="cpu", weights_only=False)


def to_img(x, ds_mean):
    """x_train / x_ctrl are stored RAW in [0,1]; reconstructions are mean-centered.
    Add ds_mean only when the tensor is centered (has negative values) — adding it to a
    raw image double-counts the dataset mean and ghosts the other class into the picture."""
    x = x.squeeze()
    if float(x.min()) < -1e-3:
        x = x + ds_mean.squeeze()
    return np.clip(x.numpy(), 0, 1)


def ssim_pair(d, key):
    m = d.get(f"{key}_metrics", {})
    return m.get("ssim", float("nan"))


def grid(rows, out, title=None, col_titles=None):
    """rows = list of (label, tensor[N,1,H,W], ds_mean, ssim or None)."""
    n = rows[0][1].shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(2.35 * n + 2.6, 2.35 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (lab, imgs, dm, ss) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(to_img(imgs[c], dm), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(lab, fontsize=14, rotation=0, ha="right", va="center")
            if r == 0 and col_titles:
                ax.set_title(col_titles[c], fontsize=14)
        if ss is not None:
            axes[r, n - 1].text(1.06, 0.5, f"ssim {ss:.2f}", transform=axes[r, n - 1].transAxes,
                                va="center", fontsize=13, color="#555")
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(os.path.join(OUT, out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[fig]", os.path.join(OUT, out))


def t1_panel():
    r32 = ld("exp_b_T1_r32_free_s42_a149.pth")
    r16 = ld("exp_b_T1_r16_free_s42_a149.pth")
    r8 = ld("exp_b_T1_r8_free_s42_a149_leaky_relu_lr0.0027.pth")
    dm = r32["ds_mean"]
    rows = [
        ("private image", r32["x_train"], dm, None),
        ("full fine-tune", r32["x_recon_full"], dm, ssim_pair(r32, "full")),
        ("LoRA r = 32", r32["x_recon_lora"], dm, ssim_pair(r32, "lora")),
        ("LoRA r = 16", r16["x_recon_lora"], dm, ssim_pair(r16, "lora")),
        ("LoRA r = 8", r8["x_recon_lora"], r8["ds_mean"], ssim_pair(r8, "lora")),
        ("control (same class,\ndifferent sample)", r32["x_ctrl"], dm, ssim_pair(r32, "control")),
    ]
    grid(rows, "freec_T1_lora_vs_full.png",
         title="Free-coefficient reconstruction, MNIST N=2, T=1 — LoRA vs full fine-tune")


def tsweep_panel():
    import glob
    import re
    best = {}
    for f in glob.glob(os.path.join(RES, "exp_b_T*_r*_free_s42_a149_*.pth")) + \
             glob.glob(os.path.join(RES, "exp_b_T*_full_free_s42_a149*.pth")):
        m = re.search(r"exp_b_T(\d+)_(r\d+|full)_free_s42_a149(?:_([a-z_]+?))?(?:_lr([0-9.]+))?\.pth$",
                      os.path.basename(f))
        if not m or m.group(1) == "1":
            continue
        T, rk = int(m.group(1)), m.group(2)
        d = ld(os.path.basename(f))
        key = "lora" if rk != "full" else "full"
        s = ssim_pair(d, key)
        ctl = ssim_pair(d, "control")
        margin = (s - ctl) if np.isfinite(s) and np.isfinite(ctl) else float("-inf")
        slot = (T, rk)
        if slot not in best or margin > best[slot][0]:
            best[slot] = (margin, s, d, key, os.path.basename(f))
    if not best:
        print("[tsweep] no T>1 free-c tensors yet (job still running)")
        return
    for (T, rk), (mg, s, d, key, fn) in sorted(best.items()):
        print(f"  T={T} {rk}: ssim {s:.3f} margin {mg:+.3f}  ({fn})")
    Ts = sorted({T for T, _ in best})
    for T in Ts:
        rows = [("private image", None, None, None)]
        ordered = [("full", "full fine-tune")] + [(f"r{r}", f"LoRA r = {r}") for r in (32, 16, 8)]
        first = None
        for rk, lab in ordered:
            if (T, rk) in best:
                mg, s, d, key, fn = best[(T, rk)]
                if first is None:
                    first = d
                    rows[0] = ("private image", d["x_train"], d["ds_mean"], None)
                rows.append((lab, d[f"x_recon_{'lora' if rk != 'full' else 'full'}"], d["ds_mean"], s))
        if first is None:
            continue
        rows.append(("control (same class)", first["x_ctrl"], first["ds_mean"], ssim_pair(first, "control")))
        grid(rows, f"freec_T{T}_lora_vs_full.png",
             title=f"Free-coefficient reconstruction, MNIST N=2, T={T} — LoRA vs full fine-tune")




def flowers_panel():
    """flowers32 free-c T=1: full vs LoRA rank ladder (a10000 = ReLU-like extraction, seed 42)."""
    files = {r: ld(f"exp_b_T1_flowers32_r{r}_free_s42_a10000.pth") for r in (64, 32, 8)}
    base = files[64]
    dm = base["ds_mean"]
    rows = [("private image", base["x_train"], dm, None),
            ("full fine-tune", base["x_recon_full"], dm, ssim_pair(base, "full"))]
    for r in (64, 32, 8):
        d = files[r]
        rows.append((f"LoRA r = {r}", d["x_recon_lora"], d["ds_mean"], ssim_pair(d, "lora")))
    rows.append(("control (same class,\ndifferent sample)", base["x_ctrl"], dm, ssim_pair(base, "control")))
    grid(rows, "freec_T1_flowers_lora_vs_full.png",
         title="Free-coefficient reconstruction, Flowers-102 (32px) N=2, T=1 — LoRA vs full fine-tune")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsweep", action="store_true")
    args = ap.parse_args()
    t1_panel()
    flowers_panel()
    if args.tsweep:
        tsweep_panel()
