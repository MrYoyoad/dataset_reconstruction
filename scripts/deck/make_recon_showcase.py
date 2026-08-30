"""Showcase figures: REALISTIC (free-coefficient) reconstructions, LoRA vs full fine-tune.

Rules baked in (user directives + project metric hygiene):
  - free-coefficients ONLY; oracle appears nowhere.
  - selection metric = ctrl_margin_norm = ssim_norm(recon) - ssim_norm(control)  (the project's
    clip-robust leakage proxy, experiments/recompute_metrics.py), NOT raw ssim alone;
  - the trivial-baseline gate is always shown: raw ssim vs ssim_mean_baseline (experiments/metrics.py:
    a result at/below the dataset-mean baseline carries NO instance-specific information);
  - per-row labels carry raw ssim, ssim_norm and the margin; a CSV of EVERY sweep cell with all
    stored metrics is written for audit (results/recon_showcase_sweep.csv).

  python scripts/deck/make_recon_showcase.py            # T=1 reference panels (not the deliverable)
  python scripts/deck/make_recon_showcase.py --tsweep   # score + render the T>1 sweep (jobs 323866/323867)
"""
import argparse
import csv as csvmod
import glob
import os
import re
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

METRIC_KEYS = ["ssim", "ssim11", "ssim_norm", "ssim_norm11", "ncc", "l2", "ssim_mean_baseline",
               "clipped_fraction"]


def ld(name):
    return torch.load(os.path.join(RES, name), map_location="cpu", weights_only=False)


def to_img(x, ds_mean):
    """x_train / x_ctrl are stored RAW in [0,1]; reconstructions are mean-centered.
    Add ds_mean only when the tensor is centered (has negative values)."""
    x = x.squeeze()
    if float(x.min()) < -1e-3:
        x = x + ds_mean.squeeze()
    x = x.numpy()
    if x.ndim == 3:  # C,H,W -> H,W,C
        x = np.transpose(x, (1, 2, 0))
    return np.clip(x, 0, 1)


def m(d, key):
    return d.get(f"{key}_metrics", {}) or {}


def per_tile(d, key, recon_key=None):
    """Per-image (raw ssim, ssim_norm), mirroring experiments/run_experiment_b.py exactly:
    recon (centered) vs x_train - ds_mean for key in {lora, full}; for key == "control" the
    RECONSTRUCTION is compared with the centered same-class control image (that is what the
    stored control_metrics are — the margin's reference), using the recon named by recon_key.
    Returns None outside the rec env (kornia missing)."""
    try:
        sys.path.insert(0, ROOT)
        sys.path.insert(0, os.path.join(ROOT, "dataset_reconstruction"))
        from experiments.metrics import compute_ssim, compute_ssim_normalized
        dm = d["ds_mean"]
        if key == "control":
            rk = recon_key or ("lora" if "x_recon_lora" in d else "full")
            rec = d[f"x_recon_{rk}"]
            tgt = d["x_ctrl"] - dm
        else:
            rec = d[f"x_recon_{key}"]
            tgt = d["x_train"] - dm
        raw, _ = compute_ssim(rec, tgt, dm)
        nrm, _ = compute_ssim_normalized(rec, tgt, dm)
        return [(float(a), float(b)) for a, b in zip(raw, nrm)]
    except Exception:
        return None


def margin_norm(d, key="lora"):
    """ctrl_margin_norm: ssim_norm(recon) - ssim_norm(same-class control)."""
    a, c = m(d, key).get("ssim_norm"), m(d, "control").get("ssim_norm")
    return (a - c) if a is not None and c is not None else float("nan")


def row_label(d, key):
    mm = m(d, key)
    s, sn = mm.get("ssim", float("nan")), mm.get("ssim_norm", float("nan"))
    return f"ssim {s:.2f} · norm {sn:.2f} · margin {margin_norm(d, key):+.2f}"


def baseline_note(d, key):
    mm = m(d, key)
    s, b = mm.get("ssim", float("nan")), mm.get("ssim_mean_baseline", float("nan"))
    ok = s > b
    return f"{'beats' if ok else 'DOES NOT beat'} the dataset-mean baseline ({s:.2f} vs {b:.2f})", ok


def grid(rows, out, title=None, note=None):
    """rows = list of (label, tensor[N,C,H,W], ds_mean, metrics_string or None)."""
    n = rows[0][1].shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(2.35 * n + 3.6, 2.35 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        lab, imgs, dm, ms = row[:4]
        tiles = row[4] if len(row) > 4 else None
        for c in range(n):
            ax = axes[r, c]
            im = to_img(imgs[c], dm)
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(lab, fontsize=14, rotation=0, ha="right", va="center")
            if tiles:
                ax.set_xlabel(f"{tiles[c][0]:.2f} / {tiles[c][1]:.2f}", fontsize=11, color="#555")
        if ms:
            axes[r, n - 1].text(1.06, 0.5, ms, transform=axes[r, n - 1].transAxes,
                                va="center", fontsize=11.5, color="#555")
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    if note:
        fig.text(0.5, -0.01, note, ha="center", fontsize=11, color="#666", style="italic")
    fig.savefig(os.path.join(OUT, out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[fig]", os.path.join(OUT, out))


# ------------------------------------------------------------------ sweep scoring
PAT = re.compile(r"exp_b_T(\d+)(?:_(flowers32|fashion))?_(r\d+|full)_free_s42_a(?:149|10000)"
                 r"(?:_([a-z_]+?))?(?:_lr([0-9.]+))?\.pth$")


def scan(min_T=2):
    cells = []
    seen = set()
    files = set(glob.glob(os.path.join(RES, "exp_b_T*_free_s42_*.pth")))
    files |= set(glob.glob(os.path.join(RES, "exp_b_T*_*_free_s42_*.pth")))
    for f in sorted(files):
        if f in seen:
            continue
        seen.add(f)
        b = os.path.basename(f)
        mo = PAT.match(b)
        if not mo:
            continue
        T = int(mo.group(1))
        if T < min_T:
            continue
        ds = mo.group(2) or "mnist"
        rk = mo.group(3)
        key = "full" if rk == "full" else "lora"
        try:
            d = ld(b)
        except Exception:
            continue
        if f"x_recon_{key}" not in d:
            continue
        row = {"file": b, "dataset": ds, "T": T, "rank": rk,
               "activation": mo.group(4) or "(default)", "lr": mo.group(5) or "(default)",
               "margin_norm": round(margin_norm(d, key), 4)}
        for k in METRIC_KEYS:
            row[k] = m(d, key).get(k)
            row[f"ctrl_{k}"] = m(d, "control").get(k)
        cells.append((row, d, key))
    return cells


def tsweep(write_csv=True):
    cells = scan()
    if not cells:
        print("[tsweep] no T>1 free-c tensors yet (jobs 323866/323867 still running)")
        return
    if write_csv:
        path = os.path.join(RES, "recon_showcase_sweep.csv")
        with open(path, "w", newline="") as fh:
            w = csvmod.DictWriter(fh, fieldnames=list(cells[0][0].keys()))
            w.writeheader()
            for row, _, _ in cells:
                w.writerow(row)
        print(f"[csv] {path}  ({len(cells)} cells)")
    # best per (dataset, T, rank) by ctrl_margin_norm
    best = {}
    for row, d, key in cells:
        slot = (row["dataset"], row["T"], row["rank"])
        if slot not in best or row["margin_norm"] > best[slot][0]["margin_norm"]:
            best[slot] = (row, d, key)
    print("\nbest cell per (dataset, T, rank) by ctrl_margin_norm:")
    for slot in sorted(best):
        row, d, key = best[slot]
        gate, ok = baseline_note(d, key)
        print(f"  {slot}: margin {row['margin_norm']:+.3f}  ssim {row['ssim']:.3f}  "
              f"norm {row['ssim_norm']:.3f}  [{gate}]  {row['file']}")
    # one grid per (dataset, T): full + rank ladder
    for ds in sorted({s[0] for s in best}):
        for T in sorted({s[1] for s in best if s[0] == ds}):
            order = [("full", "full fine-tune")] + \
                    [(f"r{r}", f"LoRA r = {r}") for r in (32, 16, 8)]
            rows, first, gates = [], None, []
            for rk, lab in order:
                if (ds, T, rk) not in best:
                    continue
                row, d, key = best[(ds, T, rk)]
                if first is None:
                    first, first_key = d, key
                    rows.append(("private image", d["x_train"], d["ds_mean"], None))
                rows.append((lab, d[f"x_recon_{key}"], d["ds_mean"], row_label(d, key), per_tile(d, key)))
                gates.append((lab, baseline_note(d, key)))
            if first is None:
                continue
            rows.append(("same-class control image\n(recon scored against it)", first["x_ctrl"], first["ds_mean"],
                         row_label(first, "control").replace(f"margin {margin_norm(first,'control'):+.2f}", ""),
                         per_tile(first, "control", first_key)))
            fails = [lab for lab, (g, ok) in gates if not ok]
            note = ("free coefficients (realistic attack) · under each tile: raw ssim / ssim_norm of the reconstruction vs that tile's image · "
                    "row label = mean; margin = ssim_norm(recon) − ssim_norm(control) · "
                    + ("all rows beat the dataset-mean baseline" if not fails
                       else "baseline gate FAILED for: " + ", ".join(fails)))
            grid(rows, f"freec_{ds}_T{T}_lora_vs_full.png",
                 title=f"Free-coefficient reconstruction, {ds} N=2, T={T} — LoRA vs full fine-tune",
                 note=note)


# ------------------------------------------------------------------ T=1 reference panels
def t1_panel():
    r32 = ld("exp_b_T1_r32_free_s42_a149.pth")
    r16 = ld("exp_b_T1_r16_free_s42_a149.pth")
    r8 = ld("exp_b_T1_r8_free_s42_a149_leaky_relu_lr0.0027.pth")
    dm = r32["ds_mean"]
    rows = [("private image", r32["x_train"], dm, None),
            ("full fine-tune", r32["x_recon_full"], dm, row_label(r32, "full")),
            ("LoRA r = 32", r32["x_recon_lora"], dm, row_label(r32, "lora")),
            ("LoRA r = 16", r16["x_recon_lora"], r16["ds_mean"], row_label(r16, "lora")),
            ("LoRA r = 8", r8["x_recon_lora"], r8["ds_mean"], row_label(r8, "lora")),
            ("control (same class,\ndifferent sample)", r32["x_ctrl"], dm, None)]
    grid(rows, "freec_T1_lora_vs_full.png",
         title="Free-coefficient reconstruction, MNIST N=2, T=1 — LoRA vs full fine-tune (reference only)",
         note="free coefficients · margin = ssim_norm(recon) − ssim_norm(control)")


def flowers_panel():
    files = {r: ld(f"exp_b_T1_flowers32_r{r}_free_s42_a10000.pth") for r in (64, 32, 8)}
    base = files[64]
    dm = base["ds_mean"]
    rows = [("private image", base["x_train"], dm, None),
            ("full fine-tune", base["x_recon_full"], dm, row_label(base, "full"))]
    for r in (64, 32, 8):
        rows.append((f"LoRA r = {r}", files[r]["x_recon_lora"], files[r]["ds_mean"], row_label(files[r], "lora")))
    rows.append(("control (same class,\ndifferent sample)", base["x_ctrl"], dm, None))
    grid(rows, "freec_T1_flowers_lora_vs_full.png",
         title="Free-coefficient reconstruction, Flowers-102 (32px) N=2, T=1 — LoRA vs full fine-tune (reference only)",
         note="free coefficients · margin = ssim_norm(recon) − ssim_norm(control)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsweep", action="store_true")
    args = ap.parse_args()
    if args.tsweep:
        tsweep()
    else:
        t1_panel()
        flowers_panel()
        tsweep()
