#!/usr/bin/env python3
"""Rebuild the added-on-class figures from saved tensors with a layout that does not collide.

The in-script figures wrote row labels with `fig.text` at a fixed x while the axes started at x=0.16, so the longest
label ran under the first image; the per-image captions sat in the gap between rows with no headroom; and the whole
configuration was crammed into one title line. Nothing is recomputed here -- every figure is redrawn from the .pth
the cell already saved, so this costs no GPU and cannot change a number.

  python -m experiments.cifar.replot_newclass [--out figures/cifar_newclass]
"""
import argparse, glob, os
import torch

ROWS = [("private image", "x_raw"),
        ("chart projection\n(the fidelity ceiling)", "x_chart"),
        ("BEST-scoring starts\n(what the attacker keeps)", "x_found_best"),
        ("the 8 WORST-scoring starts\n(what the attacker discards)", "x_found_failed")]


def rebuild_chart(blob, meta):
    """The chart is deterministic given the class and seed, so a failed start's latent can be decoded without any
       re-run. Newer cells save the chart directly; older ones are rebuilt from the same public pool and seed."""
    if "chart_mean" in blob: return blob["chart_mean"], blob["chart_V"]
    from experiments.cifar.cifar_newclass import load_cifar100_class, load_fashion, load_flowers102, load_svhn
    nc = meta["newclass"]
    if nc.startswith("cifar100:"): pool, _ = load_cifar100_class("data", nc.split(":", 1)[1])
    elif nc.startswith("fashion:"): pool, _ = load_fashion("dataset_reconstruction/data", nc.split(":", 1)[1])
    elif nc == "svhn": pool, _ = load_svhn("data")
    elif nc.startswith("svhn:"): pool, _ = load_svhn("data", nc.split(":", 1)[1])
    else: pool, _ = load_flowers102("data", seed=meta["seed"])
    Pub = torch.tensor(pool["train"], dtype=torch.float64)
    mean = Pub.mean(0); _, _, Vh = torch.linalg.svd(Pub - mean, full_matrices=False)
    return mean, Vh[: meta["k"]].T.contiguous()


def failed_starts(blob, meta, N):
    """The N WORST-scoring starts that recovered nothing -- deliberately the worst end, not a sample of failures.
       The claim being drawn is about an ORDERING, and the honest way to show an ordering is to show its two ends.
       An attacker never looks at this end; they take the top of their own ranking, which is measured clean at 20 of
       20 in every landing cell. This row is what they would be discarding."""
    runs = blob.get("runs") or []
    bad = [i for i, r in enumerate(runs) if not r.get("landed") and not r.get("degenerate")]
    if not bad or "W" not in blob: return None, None
    bad = sorted(bad, key=lambda i: -runs[i]["objective"])[:N]                   # may be FEWER than N in a cell where nearly every start lands
    mean, V = rebuild_chart(blob, meta)
    X = mean[:, None] + V @ blob["W"][:, bad].double()
    return X, [runs[i]["objective"] for i in bad]


def pretty(meta):
    """Two lines: what the cell IS, then what it did. Everything else goes in the footnote."""
    cname = meta["class_name"].replace("_", " ")
    arch = "over-trained MLP" if meta.get("overtrained") else ("MLP" if meta["arch"] == "mlp" else "CNN")
    what = f"'{cname}' added as a new class to a CIFAR-10 {arch}"
    if meta.get("wrong_release"):
        did = (f"CONTROL: the attack is given a release trained on eight DIFFERENT images — "
               f"{meta['landed']} of {meta['starts']} starts land, {meta['images_found']} of {meta['N']} images found")
    elif meta["private"] == "raw":
        did = (f"CONTROL: private images NOT on the chart — {meta['landed']} of {meta['starts']} starts land, "
               f"{meta['images_found']} of {meta['N']} images found")
    else:
        did = (f"{meta['landed']} of {meta['starts']} random starts recover a private image; "
               f"{meta['images_found']} of {meta['N']} images found")
    ck = meta.get("chart_kind", "pca")
    foot = (f"head adapter, rank {meta['r']}, {meta['T']} SGD steps · "
            f"{'ORACLE chart (spans the private images)' if ck == 'oracle' else 'public PCA chart'}, {meta['k']} dimensions · "
            f"backbone {meta['backbone_test_acc']*100:.0f}% test / {meta['backbone_train_acc']*100:.1f}% train · "
            f"certificate residual at the private images {max(meta['residual_at_truth']):.0e} against "
            f"{meta['res_public_median']:.1e} on public images")
    return what, did, foot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/cifar_newclass"); ap.add_argument("--out", default="figures/cifar_newclass")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_done = 0
    for f in sorted(glob.glob(os.path.join(a.src, "*.pth"))):
        blob = torch.load(f, map_location="cpu", weights_only=False)
        if "meta" not in blob or "x_found_best" not in blob: continue
        meta = blob["meta"]; N = meta["N"]
        Xf, obj_f = failed_starts(blob, meta, N)
        n_fail = 0 if Xf is None else Xf.shape[1]
        rows_here = list(ROWS) if n_fail else list(ROWS[:3])       # no failing start to show -> no fourth row
        blob = dict(blob)
        if n_fail: blob["x_found_failed"] = Xf
        oracle = meta.get("chart_kind") == "oracle"
        what, did, foot = pretty(meta)
        per = {p["i"]: p for p in meta["per_image"]}

        # Geometry chosen so nothing collides: a wide left gutter for the row labels, generous headroom above the
        # last row for its per-image captions, and one line of footnote at the bottom.
        cell = 1.15
        n_rows_ = len(rows_here)
        fig_w, fig_h = 2.85 + cell * N, 1.80 + cell * n_rows_ + 0.75
        fig = plt.figure(figsize=(fig_w, fig_h))
        left, right, top, bottom = 2.75 / fig_w, 1 - 0.12 / fig_w, 1 - 1.52 / fig_h, 0.52 / fig_h
        gs = fig.add_gridspec(len(rows_here), N, left=left, right=right, top=top, bottom=bottom, wspace=0.08, hspace=0.42)
        for ri, (label, key) in enumerate(rows_here):
            M = blob[key]
            if ri == 3: label = label.replace("the 8 WORST", f"the {n_fail} WORST")
            for j in range(N):
                if ri == 3 and j >= n_fail:                        # fewer failures than columns: leave the slot empty
                    ax = fig.add_subplot(gs[ri, j]); ax.axis("off")
                    if j == n_fail:
                        ax.text(0.0, 0.5, "no further\nfailing starts", fontsize=7.5, va="center", color="#888888")
                    continue
                ax = fig.add_subplot(gs[ri, j])
                ax.imshow(M[:, j].reshape(3, 32, 32).permute(1, 2, 0).clamp(0, 1).float().numpy())
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_linewidth(0.4); sp.set_color("#999999")
                if ri == 2:
                    p = per.get(j, {})
                    ok = bool(p.get("landings")) or (p.get("best_err") is not None and p["best_err"] < 1e-2)
                    txt = "recovered" if ok else (f"missed ({p['best_err']:.2f})" if p.get("best_err") is not None else "missed")
                    obj = p.get("best_objective")
                    ax.set_title(txt + (f"\nresidual {obj:.0e}" if obj else ""), fontsize=7.5, pad=3.5,
                                 color=("#1a7f37" if ok else "#b3261e"))
                if ri == 3 and obj_f:
                    ax.set_title(f"failed\nresidual {obj_f[j]:.0e}", fontsize=7.5, pad=3.5, color="#b3261e")
                if j == 0:
                    pos = ax.get_position()
                    fig.text(left - 0.014, (pos.y0 + pos.y1) / 2, label, fontsize=9.5, va="center", ha="right")
        if oracle:
            fig.text(0.5, 1 - 0.06 / fig_h,
                     "ORACLE CHART — the chart is built from the span of the PRIVATE IMAGES themselves. This is the fidelity "
                     "ceiling, not an attack: no attacker could construct this chart.",
                     fontsize=9, ha="center", va="top", color="#b3261e")
        fig.text(0.5, 1 - 0.34 / fig_h, what, fontsize=12.5, ha="center", va="top")
        fig.text(0.5, 1 - 0.74 / fig_h, did, fontsize=10, ha="center", va="top", color="#333333")
        fig.text(0.5, 1 - 1.06 / fig_h,
                 ("bottom two rows are the two ends of the attacker's OWN ranking by residual — no ground truth is used to sort them"
                  if n_fail else "every non-degenerate start recovered a private image, so there is no failing start to show"),
                 fontsize=8.5, ha="center", va="top", color="#555555")
        fig.text(0.5, 0.16 / fig_h, foot, fontsize=7.5, ha="center", va="bottom", color="#666666")
        out = os.path.join(a.out, os.path.basename(f).replace(".pth", ".png"))
        fig.savefig(out, dpi=200); plt.close(fig); n_done += 1
        print(f"# {out}", flush=True)
    print(f"# redrew {n_done} figures")


if __name__ == "__main__":
    main()
