#!/usr/bin/env python3
"""Perceptual identification tier for saved reconstruction cells (library + CLI).

The repo's first tier is EXACT LANDING: relative pixel error ||x_hat - x|| / ||x|| < 1e-2. This module adds a
SECOND, weaker tier -- is the recovered image identifiably THE private image, as a human would judge it, rather
than a generic public look-alike of the same class? -- and measures it by a DECOY LINE-UP: the truth is placed
among 99 PUBLIC images of the same class (train split; the privates come from the test split, so the pools are
disjoint by construction) and the recovery has to pick it out.

Per recovered image the tier reports
  (1) ssim_truth       SSIM(recovery, truth)                    kornia.metrics.ssim, window 3, [0,1] -- the repo's convention
  (2) ssim_control     SSIM(recovery, nearest public image of the same class to the truth, pixel L2, train split)
      ssim_truth_vs_control   SSIM(truth, that nearest public image) -- what a look-alike scores against the truth
  (3) rank_ssim, rank_l2      rank of the truth among {truth + 99 decoys} when the 100 candidates are ordered by
                              SSIM to the recovery (descending) / pixel L2 to the recovery (ascending);
                              top1_ssim / top5_ssim / top1_l2 / top5_l2
  (4) rank_feat_l2, rank_feat_cos   the same line-up in the FROZEN PUBLIC base model's penultimate features
                              (TrainedBackbone.phi for the MNIST MLPs, inputs_of(.)[-1] for the 15-layer MLP, the
                              CNN's / MLP's .phi for the CIFAR bases); top1_feat / top5_feat use the L2 rank
  (5) err_rel, landed          the exact-landing flag against the same reference image, for cross-reference
rank = 1 + number of decoys scoring STRICTLY better than the truth; decoys are the first 99 of a fixed permutation
(numpy default_rng(DECOY_SEED)) of the class's public pool, the same 99 for every image of that class.

Sources (read by `load_arms`, one "arm" = one (recovered set, reference set) pair; the formats were read off the savers):
  oracle_ladder    results/oracle_ladder/*.pth       ladder_cell.py: x_raw, x_proj, x_found_best (D,N) + row
  ntk_vs_cert      results/ntk_vs_cert/*.pth         ntk_vs_certificate.py per-cell: cell + x_raw, x_chart, ntk, cert, control (D,N);
                                                     certificate and NTK bests are each scored against the RAW truth and the ON-CHART target
  bootstrap_chart  results/bootstrap_chart/*.pth     bootstrap.py per (variant, round, arm): x_raw + x_found/x_matched/x_slots + row.per_image
  decoder_chart    results/decoder_chart/*.pth       fidelity.py: x_truth and every x_* image set, (N,D) ROWS (fp32)

  python -m experiments.utils.perceptual_id results/oracle_ladder/mlp_letter_a_eps0.03.pth [--figure]
"""
import glob, json, math, os, re, sys, time
import numpy as np
import torch

from experiments.exact_inversion.new_class import load_emnist_letters
from experiments.exact_inversion.trained_backbone import TrainedBackbone, read_idx
from experiments.exact_inversion.deep_stack import load_deep, inputs_of
from experiments.cifar.cifar_newclass import load_cifar100_class, CNN, MLP

torch.set_default_dtype(torch.float64)

LAND = 1e-2               # exact-landing threshold (ladder / RECOVER_TOL)
N_DECOYS = 99             # line-up size = truth + 99
DECOY_SEED = 0
MNIST_ROOT = "dataset_reconstruction/data"
CIFAR_ROOT = "data"
DEV = torch.device("cpu")

# image sets of experiments/decoder_chart/fidelity.py (IMAGE_SETS), copied rather than imported because that module
# switches the default dtype to float32 at import time
DECODER_IMAGE_SETS = {"mlp_motorcycle": ("cifar100", "motorcycle", "models/exact_inversion/cifar10_mlp_newclass.pth", (3, 32, 32)),
                      "cnn_keyboard":   ("cifar100", "keyboard",   "models/exact_inversion/cifar10_cnn_newclass.pth", (3, 32, 32)),
                      "mnist_letter_a": ("emnist", "a",            "models/exact_inversion/mnist_mlp_strong.pth",     (1, 28, 28))}


def log(s): print(s, flush=True)


# ------------------------------------------------------------------------------------------------ public pools (TRAIN split only)
_POOLS = {}


def public_pool(key):
    """key = ('emnist', letter) | ('mnist_digit', label) | ('cifar100', class name). (n, D) float64, TRAIN split, [0, 1]."""
    if key in _POOLS: return _POOLS[key]
    domain, cls = key
    if domain == "emnist":
        P = torch.tensor(load_emnist_letters(MNIST_ROOT, cls)["train"][0], dtype=torch.float64)
    elif domain == "mnist_digit":
        X, y = read_idx(MNIST_ROOT, "train"); P = torch.tensor(X[y == int(cls)], dtype=torch.float64)
    elif domain == "cifar100":
        pool, _ = load_cifar100_class(CIFAR_ROOT, cls); P = torch.tensor(pool["train"], dtype=torch.float64)
    else:
        raise ValueError(f"unknown public pool domain {domain}")
    _POOLS[key] = P
    log(f"#   public pool {key}: {tuple(P.shape)} (train split)")
    return P


def decoys_of(key, n=N_DECOYS, seed=DECOY_SEED):
    """The fixed line-up decoys of a class: first n of default_rng(seed).permutation(pool)."""
    P = public_pool(key)
    idx = np.random.default_rng(seed).permutation(P.shape[0])[:n]
    return P[torch.as_tensor(idx)], [int(v) for v in idx]


def class_key(dataset_or_name, class_name, label=None):
    """Map a saver's class label to a public-pool key. 'letter_a' -> ('emnist','a'); 'digits' + label -> ('mnist_digit', label);
    a CIFAR-100 fine-label name -> ('cifar100', name); 'a' with an MNIST-side dataset -> ('emnist','a')."""
    c = str(class_name)
    if c.startswith("letter_"): return ("emnist", c[len("letter_"):])
    if c.startswith("emnist_letter_"): return ("emnist", c[len("emnist_letter_"):])
    if c == "digits": return ("mnist_digit", int(label))
    if dataset_or_name in ("mnist", "emnist") and len(c) == 1 and c.isalpha(): return ("emnist", c)
    return ("cifar100", c)


# ------------------------------------------------------------------------------------------------ feature extractors (frozen public bases)
_FEATS = {}


def feature_extractor(ckpt):
    """Penultimate features of the frozen PUBLIC base, as a map (B, D) -> (B, n). Selected by the checkpoint file name."""
    if ckpt in _FEATS: return _FEATS[ckpt]
    base = os.path.basename(ckpt)
    if base.startswith("mnist_mlp_d15"):
        Ws, b1, _ = load_deep(ckpt, DEV)
        f = lambda X: inputs_of(X.T, Ws, b1)[-1].T                                   # the head's 1000-d input, as in the k-sweep
    elif base.startswith("mnist_mlp"):
        bb = TrainedBackbone(ckpt, DEV, "gelu"); f = lambda X: bb.phi(X.T).T
    elif "cnn" in base or "mlp" in base:
        net = (CNN if "cnn" in base else MLP)()
        blob = torch.load(ckpt, map_location="cpu", weights_only=False); net.load_state_dict(blob["state_dict"])
        net = net.double().eval()
        for p_ in net.parameters(): p_.requires_grad_(False)
        f = lambda X: net.phi(X)                                                       # CNN: 256-d penultimate; MLP: 1000-d
    else:
        raise ValueError(f"no feature extractor known for checkpoint {ckpt}")
    def feat(X):
        with torch.no_grad(): return f(X.to(torch.float64))
    _FEATS[ckpt] = feat
    return feat


# ------------------------------------------------------------------------------------------------ metrics
def ssim_pairs(X, Y, shape):
    """Row-wise SSIM of two (B, D) image sets: kornia.metrics.ssim, window 3, clamped to [0,1] -- common_utils/image.py's convention."""
    import kornia.metrics as km
    a = X.reshape(-1, *shape).clamp(0, 1).float(); b = Y.reshape(-1, *shape).clamp(0, 1).float()
    return km.ssim(a, b, window_size=3).reshape(a.shape[0], -1).mean(1).double()


def _rank(score_truth, score_decoys, higher_is_better):
    better = (score_decoys > score_truth) if higher_is_better else (score_decoys < score_truth)
    return int(better.sum()) + 1


def score_image(rec, ref, key, shape, feat=None):
    """The tier for ONE recovered image `rec` (D,) against its reference `ref` (D,), class pool `key`."""
    D_, idx = decoys_of(key)
    cands = torch.cat([ref[None], D_], 0)                                              # (100, D): truth first
    R = rec[None].expand(cands.shape[0], -1)
    s = ssim_pairs(R, cands, shape); l2 = torch.linalg.norm(cands - rec[None], dim=1)
    P = public_pool(key); j = int(torch.linalg.norm(P - ref[None], dim=1).argmin()); ctrl = P[j]    # nearest public image to the TRUTH
    out = dict(err_rel=float(torch.linalg.norm(rec - ref) / torch.linalg.norm(ref)),
               ssim_truth=float(s[0]), ssim_control=float(ssim_pairs(rec[None], ctrl[None], shape)[0]),
               ssim_truth_vs_control=float(ssim_pairs(ref[None], ctrl[None], shape)[0]),
               l2_truth=float(l2[0]), l2_control=float(torch.linalg.norm(rec - ctrl)), control_pool_index=j,
               rank_ssim=_rank(s[0], s[1:], True), rank_l2=_rank(l2[0], l2[1:], False),
               ssim_best_decoy=float(s[1:].max()), lineup_size=int(cands.shape[0]), decoy_seed=DECOY_SEED)
    out["landed"] = bool(out["err_rel"] < LAND)
    out["top1_ssim"] = out["rank_ssim"] == 1; out["top5_ssim"] = out["rank_ssim"] <= 5
    out["top1_l2"] = out["rank_l2"] == 1; out["top5_l2"] = out["rank_l2"] <= 5
    if feat is not None:
        F_ = feat(cands); fr = feat(rec[None])[0]
        fl2 = torch.linalg.norm(F_ - fr[None], dim=1)
        cos = (F_ @ fr) / (torch.linalg.norm(F_, dim=1) * torch.linalg.norm(fr)).clamp_min(1e-300)
        out.update(rank_feat_l2=_rank(fl2[0], fl2[1:], False), rank_feat_cos=_rank(cos[0], cos[1:], True),
                   feat_cos_truth=float(cos[0]), feat_l2_truth=float(fl2[0]), feat_l2_best_decoy=float(fl2[1:].min()))
        out["top1_feat"] = out["rank_feat_l2"] == 1; out["top5_feat"] = out["rank_feat_l2"] <= 5
    return out


def score_arm(arm):
    """All images of one arm. arm: dict(recovered (N,D), reference (N,D), keys [N class keys], shape, ckpt, ...)."""
    feat = feature_extractor(arm["ckpt"]) if arm.get("ckpt") else None
    rows = []
    for i in range(arm["recovered"].shape[0]):
        r = score_image(arm["recovered"][i].double(), arm["reference"][i].double(), arm["keys"][i], arm["shape"], feat)
        r.update(i=i, class_key=list(map(str, arm["keys"][i])))
        rows.append(r)
    return rows


def summarise(rows):
    """Per-arm counts and medians (the summary-table line)."""
    n = len(rows); med = lambda k: float(np.median([r[k] for r in rows])) if n else float("nan")
    cnt = lambda k: int(sum(bool(r.get(k)) for r in rows))
    return dict(n=n, landed=cnt("landed"), top1_ssim=cnt("top1_ssim"), top5_ssim=cnt("top5_ssim"), top1_l2=cnt("top1_l2"), top5_l2=cnt("top5_l2"),
                top1_feat=cnt("top1_feat"), top5_feat=cnt("top5_feat"), ssim_truth_median=med("ssim_truth"), ssim_control_median=med("ssim_control"),
                ssim_truth_vs_control_median=med("ssim_truth_vs_control"), err_rel_median=med("err_rel"), rank_ssim_median=med("rank_ssim"),
                rank_feat_l2_median=med("rank_feat_l2") if rows and "rank_feat_l2" in rows[0] else float("nan"))


# ------------------------------------------------------------------------------------------------ source loaders -> arms
def _T(x): return torch.as_tensor(x).detach().cpu().double()


def shape_of(D):
    """Image shape from the flattened dimension: 784 -> (1,28,28), 3072 -> (3,32,32)."""
    return {784: (1, 28, 28), 3072: (3, 32, 32)}[int(D)]


def _arms_oracle_ladder(path, d):
    row = d["row"]; N = d["x_raw"].shape[1]; D = d["x_raw"].shape[0]; g = row.get
    # the 2026-09-07 CIFAR ladder rows (ladder_cell @ 047298f) predate the shape / dataset / backbone_ckpt fields: infer them
    shape = tuple(row["shape"]) if "shape" in row else shape_of(D)
    dataset = g("dataset") or ("mnist" if D == 784 else "cifar")
    ckpt = g("backbone_ckpt")
    if ckpt is None:
        from experiments.oracle_ladder.ladder_cell import EXAMPLES
        ckpt = EXAMPLES[row["example"]]["ckpt"]
    labels = g("labels_of_privates") or [None] * N
    keys = [class_key(dataset, row["class_name"], labels[i]) for i in range(N)]
    meta = dict(example=row["example"], chart=row["chart"], eps=g("eps"), wrong_release=g("wrong_release", False), attacker_available_chart=g("attacker_available"),
                proj_err_mean=g("proj_err_mean"), proj_err_max=g("proj_err_max"), landed_starts=g("landed"), images_found=g("images_found"), k=g("k"), T=g("T"),
                class_name=row["class_name"], dataset=dataset)
    common = dict(source="oracle_ladder", cell=os.path.splitext(os.path.basename(path))[0], path=path, shape=shape, ckpt=ckpt, keys=keys, meta=meta)
    truth = _T(d["x_raw"]).T
    return [dict(common, arm="cert_best", target="raw", attacker_output=True, recovered=_T(d["x_found_best"]).T, reference=truth),
            dict(common, arm="chart_projection", target="raw", attacker_output=False, recovered=_T(d["x_proj"]).T, reference=truth)]   # the chart's ceiling, not an attack


def _arm_from_panel(path, cell, pn, cell_id):
    shape = (1, 28, 28) if cell["dataset"] == "mnist" else (3, 32, 32)
    names = pn.get("cls_name") or [cell["class_name"]] * _T(pn["x_raw"]).shape[1]
    keys = [class_key(cell["dataset"], nm) for nm in names]
    binfo = cell.get("backbone") or {}
    ckpt = binfo.get("ckpt") if isinstance(binfo, dict) else None
    meta = dict(dataset=cell["dataset"], class_name=cell["class_name"], chart=cell["chart"], k=cell["k"], T=cell["T"], chart_repr_err_median=cell.get("chart_repr_err_median"),
                cert_images_found=cell.get("cert_images_found"), ntk_images_found=cell.get("ntk_images_found"), ntk_main_form=cell.get("ntk_main_form"),
                same_row=cell.get("same_row"), backbone=(binfo.get("backbone") if isinstance(binfo, dict) else None))
    common = dict(source="ntk_vs_cert", cell=cell_id, path=path, shape=shape, ckpt=ckpt, keys=keys, meta=meta)
    raw, on = _T(pn["x_raw"]).T, _T(pn["x_chart"]).T
    arms = []
    for arm, key in (("cert_best", "cert"), ("ntk_best", "ntk")):
        if key not in pn: continue
        X = _T(pn[key]).T
        arms.append(dict(common, arm=arm, target="raw", attacker_output=True, recovered=X, reference=raw))
        arms.append(dict(common, arm=arm, target="onchart", attacker_output=True, recovered=X, reference=on))
    if "control" in pn: arms.append(dict(common, arm="control_public_nn", target="raw", attacker_output=False, recovered=_T(pn["control"]).T, reference=raw))
    arms.append(dict(common, arm="chart_projection", target="raw", attacker_output=False, recovered=on, reference=raw))
    return arms


def _arms_ntk_vs_cert(path, d):
    stem = os.path.splitext(os.path.basename(path))[0]
    if "cell" in d and "x_raw" in d: return _arm_from_panel(path, d["cell"], d, stem)
    if "panels" in d and "cells" in d:                                              # aggregate file: one panel per (chart, k, T)
        arms = []
        for c in d["cells"]:
            pn = d["panels"].get((c["chart"], c["k"], c["T"]))
            if pn is None: continue
            arms += _arm_from_panel(path, c, pn, f"{stem}__{c['chart']}_k{c['k']}_T{c['T']}")
        return arms
    log(f"#   SKIP {path}: unrecognised ntk_vs_cert format, keys {sorted(k for k in d.keys() if isinstance(k, str))[:20]}")
    return []


def _arms_bootstrap(path, d):
    row = d["row"]; shape = (1, 28, 28) if row["release"] == "mnist" else (3, 32, 32)
    key = ("emnist", row["true_class"]) if row["release"] == "mnist" else ("cifar100", row["true_class"])
    truth = _T(d["x_raw"]).T; N = truth.shape[0]; per = row["per_image"]
    meta = dict(release=row["release"], variant=row["variant"], round=row["round"], arm=row["arm"], chart=row.get("chart"), chart_class=row.get("chart_class"),
                attacker_available_chart=row.get("attacker_available", True), chart_err_median=row.get("chart_err_median"), err_median=row.get("err_median"),
                landed=row.get("landed"), reached_opt=row.get("reached_opt"), void_round0=row.get("void_round0"), void_variant_A=row.get("void_variant_A"), job=row.get("job"))
    common = dict(source="bootstrap_chart", cell=os.path.splitext(os.path.basename(path))[0], path=path, shape=shape, ckpt=row["backbone"], keys=[key] * N, meta=meta)
    arms = []
    if row["variant"] == "round0":
        Xf = _T(d["x_found"]).T
        arms.append(dict(common, arm="matched_candidate", target="raw", attacker_output=True, recovered=Xf[[p["matched_start"] for p in per]], reference=truth))
        arms.append(dict(common, arm="best_any_start", target="raw", attacker_output=False, recovered=Xf[[p["best_any_start"] for p in per]], reference=truth))   # oracle-side selection
    elif row["variant"] == "A":
        arms.append(dict(common, arm="matched_candidate", target="raw", attacker_output=True, recovered=_T(d["x_matched"]).T, reference=truth))
    elif row["variant"] == "B":
        Xs = _T(d["x_slots"]).T
        arms.append(dict(common, arm="matched_slot", target="raw", attacker_output=True, recovered=Xs[[p["matched_slot"] for p in per]], reference=truth))
    if "x_opt" in d: arms.append(dict(common, arm="chart_optimum_oracle_start", target="raw", attacker_output=False, recovered=_T(d["x_opt"]).T, reference=truth))
    if "x_chart" in d: arms.append(dict(common, arm="chart_projection", target="raw", attacker_output=False, recovered=_T(d["x_chart"]).T, reference=truth))
    return arms


def _arms_decoder(path, d):
    name = d["image_set"]; dom, cls, ckpt, shape = DECODER_IMAGE_SETS[name]
    truth = _T(d["x_truth"]); N = truth.shape[0]
    common = dict(source="decoder_chart", cell=os.path.splitext(os.path.basename(path))[0], path=path, shape=shape, ckpt=ckpt, keys=[(dom, cls)] * N,
                  meta=dict(image_set=name, indices=d.get("indices"), ks=d.get("args", {}).get("ks"), Ks=d.get("args", {}).get("Ks")))
    arms = []
    for k in d:
        if not (k.startswith("x_") and k != "x_truth" and torch.is_tensor(d[k])): continue
        avail = not (("truth_nn" in k) or ("truth_latent" in k) or k.startswith("x_ae_"))          # anchors built from the private image are not attacker-available
        arms.append(dict(common, arm=k[2:], target="raw", attacker_output=avail, recovered=_T(d[k]), reference=truth))
    return arms


SOURCES = {"oracle_ladder": ("results/oracle_ladder/*.pth", _arms_oracle_ladder),
           "ntk_vs_cert": ("results/ntk_vs_cert/*.pth", _arms_ntk_vs_cert),
           "bootstrap_chart": ("results/bootstrap_chart/*.pth", _arms_bootstrap),
           "decoder_chart": ("results/decoder_chart/*.pth", _arms_decoder)}


def detect_source(path, d):
    if "row" in d and "x_found_best" in d: return "oracle_ladder"
    if "row" in d and d["row"].get("part") == "bootstrap_chart": return "bootstrap_chart"
    if "image_set" in d and "x_truth" in d: return "decoder_chart"
    if ("cell" in d and "x_raw" in d) or "panels" in d: return "ntk_vs_cert"
    return None


def load_arms(path, source=None):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(d, dict): log(f"#   SKIP {path}: not a dict"); return []
    source = source or detect_source(path, d)
    if source is None:
        log(f"#   SKIP {path}: unrecognised format, keys {sorted(k for k in d if isinstance(k, str))[:20]}"); return []
    if source == "bootstrap_chart" and "row" not in d: log(f"#   SKIP {path}: bootstrap file without a row (classifier cache)"); return []
    return SOURCES[source][1](path, d)


# ------------------------------------------------------------------------------------------------ figure + CLI
def lineup_figure(arm, rows, out_png, n_show=3):
    """truth | recovery | control (nearest public to truth) | top-n decoys by SSIM to the recovery, one row per image."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    N = arm["recovered"].shape[0]; shape = arm["shape"]; cols = 3 + n_show
    fig, ax = plt.subplots(N, cols, figsize=(1.3 * cols + 0.5, 1.35 * N + 0.8), squeeze=False)
    for i in range(N):
        D_, _ = decoys_of(arm["keys"][i]); rec, ref = arm["recovered"][i], arm["reference"][i]
        s = ssim_pairs(rec[None].expand(D_.shape[0], -1), D_, shape); top = torch.argsort(s, descending=True)[:n_show]
        P = public_pool(arm["keys"][i]); ctrl = P[rows[i]["control_pool_index"]]
        imgs = [("truth", ref), (f"recovery\nssim {rows[i]['ssim_truth']:.2f} rank {rows[i]['rank_ssim']}", rec), (f"control\nssim {rows[i]['ssim_control']:.2f}", ctrl)] + \
               [(f"decoy #{j+1}\nssim {float(s[t]):.2f}", D_[t]) for j, t in enumerate(top.tolist())]
        for c, (lab, im) in enumerate(imgs):
            a_ = ax[i, c]; a_.axis("off"); im = im.reshape(*shape).clamp(0, 1).float()
            a_.imshow(im.permute(1, 2, 0).numpy() if shape[0] == 3 else im[0].numpy(), cmap=None if shape[0] == 3 else "gray", vmin=0, vmax=1)
            if i == 0 or c in (1, 2, 3): a_.set_title(lab, fontsize=5.5)
    fig.suptitle(f"{arm['source']} / {arm['cell']} / {arm['arm']} vs {arm['target']}", fontsize=7); fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+"); ap.add_argument("--source", default=None, choices=list(SOURCES))
    ap.add_argument("--figure", action="store_true", help="write a line-up grid per arm to figures/perceptual_id/")
    a = ap.parse_args()
    for p in a.paths:
        for arm in load_arms(p, a.source):
            rows = score_arm(arm); sm = summarise(rows)
            log(f"\n== {arm['source']} {arm['cell']} [{arm['arm']} vs {arm['target']}]  landed {sm['landed']}/{sm['n']}  SSIM top-1 {sm['top1_ssim']}/{sm['n']} top-5 {sm['top5_ssim']}/{sm['n']}  "
                f"feat top-1 {sm['top1_feat']}/{sm['n']}  median SSIM truth {sm['ssim_truth_median']:.3f} / control {sm['ssim_control_median']:.3f}")
            for r in rows:
                log(f"   i={r['i']} err {r['err_rel']:.3e} landed={int(r['landed'])}  ssim truth {r['ssim_truth']:.3f} ctrl {r['ssim_control']:.3f} (truth-vs-ctrl {r['ssim_truth_vs_control']:.3f})  "
                    f"rank ssim {r['rank_ssim']:3d} l2 {r['rank_l2']:3d} feat-l2 {r.get('rank_feat_l2', '-'):>3} feat-cos {r.get('rank_feat_cos', '-'):>3}")
            if a.figure: lineup_figure(arm, rows, os.path.join("figures/perceptual_id", f"{arm['source']}_{arm['cell']}_{arm['arm']}_{arm['target']}.png"))


if __name__ == "__main__":
    main()
