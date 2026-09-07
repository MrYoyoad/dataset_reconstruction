#!/usr/bin/env python3
"""Rebuild the CIFAR cell grids from the saved release_and_search.pt files, with readable row labels.

The in-script figure writes the row name inside the first panel's title area, where it collides with the per-image
annotation (seen on job 277370). This regenerates every grid from saved tensors -- no GPU, no re-run, no edit to a
script a running job holds -- and writes a summary table of all cells.

  python -m experiments.cifar.replot_grids [--dirs experiments/cifar/charts/*] [--out figures/cifar_charts]
"""
import argparse, glob, json, os
import torch


def label_of(r):
    layer = {1: "pixel layer", 2: "hidden layer", 3: "head (features)"}[r["layer"]]
    chart = {"pca": "public PCA", "ae": "conv-AE decoder", "oracle": f"oracle chart (eps={r.get('eps', 0)})"}[r["chart"]]
    priv = "on-chart privates" if r["private"] == "onchart" else "raw privates"
    ctrl = ", WRONG-RELEASE CONTROL" if r.get("wrong_release") else ""
    return f"LoRA on the {layer}, {chart} k={r['k']}, {priv}{ctrl}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["experiments/cifar/charts/*"])
    ap.add_argument("--out", default="figures/cifar_charts")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = []
    for d in sorted(set(sum([glob.glob(p) for p in a.dirs], []))):
        f = os.path.join(d, "release_and_search.pt")
        if not os.path.exists(f): print(f"# {d}: no tensors"); continue
        blob = torch.load(f, map_location="cpu", weights_only=False); r = blob["result"]; N = r["N"]
        X_raw, X_train, X_found = blob["H_raw"].T, blob["X_train"], blob["X_found"]
        err = torch.stack([(X_found - X_train[i:i + 1]).norm(dim=1) / X_train[i:i + 1].norm() for i in range(N)], 1)
        best = err.argmin(0)
        panels = [("private (raw image)", X_raw), ("training input" if r["private"] == "onchart" else "chart projection of the truth", blob["H_proj"].T if blob["H_proj"].shape[0] != 3072 else blob["H_proj"]),
                  ("closest random start", X_found[best].T)]
        fig, ax = plt.subplots(3, N, figsize=(1.35 * N + 1.6, 4.9))
        fig.subplots_adjust(left=0.14, top=0.84, bottom=0.02, hspace=0.32)
        for ri, (name, imgs) in enumerate(panels):
            M = imgs if imgs.shape[0] == 3072 else imgs.T
            for j in range(N):
                ax[ri, j].imshow(M[:, j].reshape(3, 32, 32).permute(1, 2, 0).clamp(0, 1).float().numpy()); ax[ri, j].axis("off")
                if ri == 2:
                    e = float(err[best[j], j])
                    ax[ri, j].set_title(("landed" if e < 1e-2 else f"err {e:.2f}") + f"\nssim {r['best_ssim_vs_raw'][j]:.2f}", fontsize=6.5)
            pos = ax[ri, 0].get_position()
            fig.text(0.012, (pos.y0 + pos.y1) / 2, name, fontsize=8, va="center", ha="left")
        fig.suptitle(f"{label_of(r)}, T={r.get('T', '?')} SGD steps (lr={r.get('lr', '?')})\n{r['landed']}/{r['starts']} starts landed, {r['images_found']}/{N} images found; "
                     f"residual at the truths {max(r['residual_at_truth']):.0e}, best start {r['residual_min']:.0e}, median start {r['residual_median']:.0e}", fontsize=9)
        p = os.path.join(a.out, os.path.basename(d) + ".png"); fig.savefig(p, dpi=150); plt.close(fig)
        rows.append((os.path.basename(d), r)); print(f"# {p}")
    # ORACLE CELLS ARE NEVER POOLED WITH ATTACK CELLS, and the split is enforced here rather than remembered by
    # whoever reads the table: an oracle chart is built from the private images, so its landings and images-found are
    # a fidelity ceiling. Once summed into a total with attack cells, no label on any figure can undo it.
    is_oracle = lambda r: r.get("chart") == "oracle"
    attack_rows = [(n, r) for n, r in rows if not is_oracle(r)]
    oracle_rows = [(n, r) for n, r in rows if is_oracle(r)]
    hdr = "| cell | landed / starts | images found | residual at truths | best start | median start | top-20 by residual landed | SSIM vs raw (attack) | SSIM vs raw (chart floor) | SSIM vs raw (control) |"
    tab = [hdr, "|---|---|---|---|---|---|---|---|---|---|"]
    mean = lambda v: sum(v) / len(v)
    for n, r in attack_rows:
        tab.append(f"| {label_of(r)} | {r['landed']}/{r['starts']} | {r['images_found']}/{r['N']} | {max(r['residual_at_truth']):.1e} | {r['residual_min']:.1e} | {r['residual_median']:.1e} | "
                   f"{sum(r['top20_by_residual_landed'])}/20 | {mean(r['best_ssim_vs_raw']):.2f} | {mean(r.get('chart_floor_ssim_vs_raw', r['chart_floor_ssim'])):.2f} | {mean(r['control_ssim']):.2f} |")
    if oracle_rows:
        tab += ["", "### NOT ATTACKER-AVAILABLE — oracle charts, built from the span of the private images.",
                "These are FIDELITY CEILINGS, not attacks. They are listed separately because they must never be pooled",
                "with the rows above, and no total or median in this file mixes them.", "", hdr,
                "|---|---|---|---|---|---|---|---|---|---|"]
        for n, r in oracle_rows:
            tab.append(f"| {label_of(r)} | {r['landed']}/{r['starts']} | {r['images_found']}/{r['N']} | {max(r['residual_at_truth']):.1e} | {r['residual_min']:.1e} | {r['residual_median']:.1e} | "
                       f"{sum(r['top20_by_residual_landed'])}/20 | {mean(r['best_ssim_vs_raw']):.2f} | {mean(r.get('chart_floor_ssim_vs_raw', r['chart_floor_ssim'])):.2f} | {mean(r['control_ssim']):.2f} |")
    open(os.path.join(a.out, "table.md"), "w").write("\n".join(tab) + "\n")
    print("\n".join(tab))


if __name__ == "__main__":
    main()
