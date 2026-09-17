#!/usr/bin/env python3
"""HEAD-TO-HEAD on the SAME release: the NTK/linearised reconstruction against the certificate.

Both attacks see exactly the same thing -- the same fine-tuned adapter on the same private images, the same public
base model, the same chart, the same random starts, the same budget, the same landing criterion. The only difference
is the equation each one solves. Swept over CHART TYPE, CHART DIMENSION k, and the number of fine-tuning STEPS T.

Setting: N private images of a class the base model does not have (CIFAR: CIFAR-100 classes such as keyboard / apple
/ motorcycle / bottle; MNIST: EMNIST letters a / t), each added class getting its own new output row (or, with
--same-row, ONE new row for both classes' images), LoRA of rank r on the HEAD, B_0 = 0, vanilla full-batch SGD,
float64. The private training inputs are their own chart projections, so both methods have a reachable target (the
on-chart setting).

THE EQUATIONS
  NTK / linearised, FREE COEFFICIENTS (Experiment B's realistic mode). TWO FORMS, because the naive one is a
  strawman on a LoRA release and the comparison has to use the strongest fair version:

    --ntk-forms dW    the ORIGINAL full-weight form: Delta W ~= sum_i r_i phi(x_i)^T, fitted against the merged
                      Delta W = B_T A_T. This is Experiment B verbatim, and it is what an attacker who sees only a
                      MERGED release can write. It is mis-specified for LoRA even at T = 1, because one LoRA step
                      moves B by -eta D (A_0 H)^T, i.e. the full gradient composed with the adapter, not the
                      gradient; the measured mis-fit at T = 1 is large for exactly that reason.
    --ntk-forms lora  the LoRA-AWARE form, and the fair one: at T = 1 the released A_T IS A_0 (B_0 = 0 kills the A
                      gradient), so the attacker can write the model in the released factors directly,
                            B_T  ~=  sum_i r_i (A_T phi(x_i))^T          (r_i FREE in R^m)
                      and fit the released B_T. At T = 1 this is EXACT up to the free coefficients, so the
                      linearised route is being run inside its own regime rather than outside it.

  Both forms minimise their residual jointly over the N latents z_i and the N coefficient vectors r_i.

  Certificate.  C = P_{row(B_T)^perp} A_T, computed from the released factors alone, and the attack minimises, per
  image and independently, ||C phi(G(z))|| / ||A_T phi(G(z))||.

Reported per cell and PER FORM: the MODEL FLOOR AT THE TRUTH, defined as the best the linearised model can do when
handed the true private images and allowed its best coefficients,
      floor  =  min_R  || target - model(R, H_true) ||_F / || target ||_F      (least squares in R, closed form)
This is the honest floor for the FREE-coefficient attack -- it is that attack restricted to the true images -- and
it separates "the model cannot express this release" from "the search did not find the images".

WHY THE COMPARISON IS INFORMATIVE. The NTK loss couples all N images through one sum, so a start returns a whole
set and any recombination reproducing that sum is a minimiser (the superposition problem). The certificate is
separable: a start returns ONE image and the other N-1 never enter.

Both are optimised by Adam, batched over restarts, with identical starts and iteration counts, so nothing is a
solver artefact; the certificate's native Levenberg-Marquardt solver runs as a third arm for reference.

CHARTS ON A TWO-CLASS CELL (WP2, 2026-09-18, audit-fixed naming). Two scopes, recorded in every row as `chart_scope`:
    pca / ae                   POOLED (unchanged from every earlier row): ONE chart fitted on the union of both classes'
                               public pools, shared by every private slot.
    pca_perclass / ae_perclass PER-CLASS (the addition): one chart per added class, fitted on that class's public pool;
                               each private slot uses its own class's chart (the attacker is told the slot classes).
Because the privates are ON-CHART, the two scopes define DIFFERENT private images and hence DIFFERENT releases:
recovery counts are compared only WITHIN a chart; charts are compared only on the RAW-image projection error (the
true held-out test images before projection; per image, median and range, printed per chart and k in the same job).
On a single-class cell the two scopes coincide and *_perclass names are skipped.

CONTROL (release-free baseline), one per private image: the nearest PUBLIC image of the chart's own fitting pool,
projected on the chart, at the same relative-error metric as the recoveries. A recovery that is not closer to the
truth than this control is no better than a guess that never saw the release.

BASE MODELS (WP0). --backbone selects the CIFAR base: cnn (cifar10_cnn_newclass.pth), mlp_overtrained
(cifar10_mlp_overtrained_newclass.pth) or mlp_weak (cifar10_mlp_newclass.pth, the 58% pixel MLP; legacy rows only).
Every row records the checkpoint path, the stored accuracies, and train accuracy / train loss / test accuracy
MEASURED AT LOAD on the full train and test splits, plus the WP0 gate (train acc >= 99.5%, train loss <= 1e-2).

  python -u -m experiments.cifar.ntk_vs_certificate --dataset cifar --backbone cnn --newclass cifar100:motorcycle
  python -u -m experiments.cifar.ntk_vs_certificate --dataset cifar --backbone cnn --newclass cifar100:motorcycle --newclass2 cifar100:bottle --charts pca ae pca_perclass
  python -u -m experiments.cifar.ntk_vs_certificate --dataset mnist --letter a --letter2 t --same-row
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_default_dtype(torch.float64)

FULLY_TRAINED_TRAIN_ACC, FULLY_TRAINED_TRAIN_LOSS = 0.995, 1e-2          # WP0 gate (plan 2026-09-18)
CIFAR_BACKBONES = {"cnn": ("models/exact_inversion/cifar10_cnn_newclass.pth", "cnn"),
                   "mlp_overtrained": ("models/exact_inversion/cifar10_mlp_overtrained_newclass.pth", "mlp"),
                   "mlp_weak": ("models/exact_inversion/cifar10_mlp_newclass.pth", "mlp")}


def log(s): print(s, flush=True)


def _acc_loss(logits_fn, X, y, dev, chunk=2000):
    """Accuracy and mean cross-entropy of a (B, D) float array under logits_fn: (B, D) tensor -> (B, m) logits."""
    n_ok, loss_sum = 0, 0.0
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            xb = torch.tensor(X[i:i + chunk], dtype=torch.float64, device=dev); yb = torch.tensor(y[i:i + chunk], device=dev)
            z = logits_fn(xb); n_ok += int((z.argmax(1) == yb).sum()); loss_sum += float(F.cross_entropy(z, yb, reduction="sum"))
    return n_ok / len(X), loss_sum / len(X)


def _gate_line(info):
    ok = info["train_acc"] >= FULLY_TRAINED_TRAIN_ACC and info["train_loss"] <= FULLY_TRAINED_TRAIN_LOSS
    info["fully_trained_gate"] = "PASS" if ok else "FAIL"
    info["fully_trained_gate_def"] = f"train acc >= {FULLY_TRAINED_TRAIN_ACC} and train loss <= {FULLY_TRAINED_TRAIN_LOSS} (WP0)"
    return (f"# base {info['ckpt']}: MEASURED AT LOAD train {info['train_acc']*100:.2f}% (loss {info['train_loss']:.2e}), "
            f"test {info['test_acc']*100:.2f}%; stored train {info.get('train_acc_stored')}, test {info.get('test_acc_stored')}, "
            f"loss {info.get('train_loss_stored')}; WP0 fully-trained gate {info['fully_trained_gate']}")


# ------------------------------------------------------------------------------------------------ backbones
def _pool_cifar(a, spec):
    from experiments.cifar.cifar_newclass import load_cifar100_class, load_flowers102
    if spec.startswith("cifar100:"): return load_cifar100_class(a.data_root, spec.split(":", 1)[1])
    return load_flowers102(a.data_root, seed=a.seed)


def build_cifar(a, dev):
    """A CIFAR-10 base (see CIFAR_BACKBONES), head extended by one zero row per added class (or one row with --same-row)."""
    from experiments.cifar.cifar_newclass import train_backbone, load_cifar10
    path, arch = CIFAR_BACKBONES[a.backbone]; path = a.ckpt or path
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: bases are trained and gated OUTSIDE this harness (WP0); refusing to train one here")
    net, te_st, tr_st = train_backbone(path, a.data_root, dev, 0, 0.0, 0.0, arch)          # loads only (path exists)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = net.double()
    for p_ in net.parameters(): p_.requires_grad_(False)
    Xtr, ytr, Xte, yte = load_cifar10(a.data_root)
    tr_acc, tr_loss = _acc_loss(lambda xb: net(xb), Xtr, ytr, dev); te_acc, te_loss = _acc_loss(lambda xb: net(xb), Xte, yte, dev)
    info = dict(ckpt=path, backbone=a.backbone, arch=arch, test_acc_stored=blob.get("test_acc"), train_acc_stored=blob.get("train_acc"),
                train_loss_stored=blob.get("train_loss"), epochs_stored=blob.get("epochs"), overtrain_stored=blob.get("overtrain"),
                train_acc=tr_acc, train_loss=tr_loss, test_acc=te_acc, test_loss=te_loss, measured_on="CIFAR-10 full train (50k) / test (10k), float64")
    log(_gate_line(info))
    if a.backbone == "mlp_weak": log("# WARNING: mlp_weak is the 58%-test pixel MLP kept for the pre-2026-09-18 rows only; it is NOT a WP2 base")
    specs = [a.newclass] + ([a.newclass2] if a.newclass2 else [])
    pools = [_pool_cifar(a, sp) for sp in specs]
    Pub = [torch.tensor(p["train"], dtype=torch.float64, device=dev) for p, _ in pools]
    Pri = [torch.tensor(p["test"], dtype=torch.float64, device=dev) for p, _ in pools]
    n_rows = 1 if a.same_row else len(specs)
    W0 = torch.cat([net.head.weight.double(), torch.zeros(n_rows, net.head.weight.shape[1], dtype=torch.float64, device=dev)], 0)
    return (lambda X: net.phi(X.T).T), W0, Pub, Pri, "+".join(nm for _, nm in pools), info, (3, 32, 32), a.backbone


def build_mnist(a, dev):
    """The repo's strong MNIST MLP with EMNIST letters as new classes (the cell of RESULTS Step 25)."""
    from experiments.exact_inversion.trained_backbone import TrainedBackbone, read_idx
    from experiments.exact_inversion.new_class import load_emnist_letters, ExtendedHead
    if not os.path.exists(a.model_mnist): raise FileNotFoundError(a.model_mnist)
    base = TrainedBackbone(a.model_mnist, dev, "gelu")
    blob = torch.load(a.model_mnist, map_location="cpu", weights_only=False)
    Xtr, ytr = read_idx(a.mnist_root, "train"); Xte, yte = read_idx(a.mnist_root, "test")
    lf = lambda xb: base.logits(xb.T).T
    tr_acc, tr_loss = _acc_loss(lf, Xtr, ytr, dev); te_acc, te_loss = _acc_loss(lf, Xte, yte, dev)
    info = dict(ckpt=a.model_mnist, backbone=os.path.splitext(os.path.basename(a.model_mnist))[0], arch="mlp 784-1000-1000-10 gelu",
                test_acc_stored=blob.get("test_acc"), train_acc_stored=blob.get("train_acc", "not stored in checkpoint"),
                train_loss_stored=blob.get("train_loss", "not stored in checkpoint"), epochs_stored=blob.get("epoch"),
                train_acc=tr_acc, train_loss=tr_loss, test_acc=te_acc, test_loss=te_loss, measured_on="MNIST full train (60k) / test (10k), float64")
    log(_gate_line(info))
    letters = [a.letter] + ([a.letter2] if a.letter2 else [])
    fls = [load_emnist_letters(a.mnist_root, L) for L in letters]
    Pub = [torch.tensor(f["train"][0], device=dev) for f in fls]; Pri = [torch.tensor(f["test"][0], device=dev) for f in fls]
    bb = ExtendedHead(base, "zero", a.seed)
    n_rows = 1 if a.same_row else len(letters)
    W0 = torch.cat([base.W0, torch.zeros(n_rows, base.W0.shape[1], device=dev)], 0)
    bb.W0 = W0; bb.m = W0.shape[0]
    return (lambda X: bb.phi(X)), W0, Pub, Pri, "+".join(f"letter_{L}" for L in letters), info, (1, 28, 28), info["backbone"]


def emnist_case_of(root, imgs, letters):
    """EMNIST 'letters' MERGES upper and lower case. Recover each private image's case by an EXACT byte match against
    the 'byclass' split (train + test), whose labels keep the case (10..35 = A..Z, 36..61 = a..z). `imgs` is (N, 784)
    in [0, 1] in MNIST orientation (as load_emnist_letters returns). Returns per image 'upper' / 'lower' / 'unknown'
    (no byte-identical byclass image) / 'ambiguous' (matches of both cases), and the matched byclass character."""
    d = os.path.join(root, "EMNIST", "raw")
    q = np.rint(np.asarray(imgs) * 255.0).astype(np.uint8).reshape(-1, 784)
    R = np.random.default_rng(0).integers(1, 2 ** 62, size=784, dtype=np.uint64)
    hq = (q.astype(np.uint64) * R).sum(1); qset = {int(h): i for i, h in enumerate(hq.tolist())}
    hits = [[] for _ in range(len(q))]
    for split in ("train", "test"):
        with open(os.path.join(d, f"emnist-byclass-{split}-labels-idx1-ubyte"), "rb") as f:
            f.read(8); lab = np.frombuffer(f.read(), dtype=np.uint8)
        with open(os.path.join(d, f"emnist-byclass-{split}-images-idx3-ubyte"), "rb") as f:
            f.read(16); raw = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
        for s in range(0, len(raw), 50000):
            blk = raw[s:s + 50000].transpose(0, 2, 1).reshape(-1, 784)
            hb = (blk.astype(np.uint64) * R).sum(1)
            for j in np.nonzero(np.isin(hb, hq))[0].tolist():
                i = qset.get(int(hb[j]))
                if i is not None and np.array_equal(blk[j], q[i]): hits[i].append(int(lab[s + j]))
    out = []
    for i, hs in enumerate(hits):
        chars = sorted({chr(65 + l - 10) if 10 <= l <= 35 else chr(97 + l - 36) if 36 <= l <= 61 else str(l) for l in hs})
        cases = {("upper" if c.isupper() else "lower" if c.islower() else "digit") for c in chars}
        case = "unknown" if not hs else (cases.pop() if len(cases) == 1 else "ambiguous")
        out.append(dict(case=case, byclass_chars="".join(chars), n_matches=len(hs)))
    return out


# ------------------------------------------------------------------------------------------------ charts
class PCAChartLocal:
    """Linear chart: the top-k principal components of the PUBLIC images it is fitted on."""
    kind = "pca"
    def __init__(s, Pub, k, dev, shape, epochs=0):
        s.mean = Pub.mean(0); U, S, Vh = torch.linalg.svd(Pub - s.mean, full_matrices=False)
        s.V = Vh[:k].T.contiguous(); s.k = k; s.explained = float((S[:k] ** 2).sum() / (S ** 2).sum()); s.n_pub = int(Pub.shape[0])
    def psi(s, Z): return s.mean[:, None] + s.V @ Z                                    # (k, P) -> (D, P)
    def psi_batch(s, Z): return s.mean[None, :, None] + torch.einsum("dk,pkn->pdn", s.V, Z)
    def coords(s, X): return s.V.T @ (X - s.mean[:, None])
    def std(s, Pub): return s.coords(Pub[:5000].T).std(dim=1, keepdim=True)
    def describe(s): return f"public PCA k={s.k} on {s.n_pub} images (explains {s.explained:.2f} of variance)"


class AEChartLocal(nn.Module):
    """Nonlinear chart: the decoder of an autoencoder trained on the PUBLIC images it is fitted on."""
    kind = "ae"
    def __init__(s, Pub, k, dev, shape, epochs=150):
        super().__init__()
        D = int(np.prod(shape)); s.k = k; s.shape = shape; s.n_pub = int(Pub.shape[0])
        h = 1024
        s.enc = nn.Sequential(nn.Linear(D, h), nn.GELU(), nn.Linear(h, 256), nn.GELU(), nn.Linear(256, k)).to(dev).double()
        s.dec = nn.Sequential(nn.Linear(k, 256), nn.GELU(), nn.Linear(256, h), nn.GELU(), nn.Linear(h, D), nn.Sigmoid()).to(dev).double()
        opt = torch.optim.Adam(list(s.enc.parameters()) + list(s.dec.parameters()), 1e-3)
        for ep in range(epochs):
            for i in torch.randperm(len(Pub), device=dev).split(256):
                xb = Pub[i]; opt.zero_grad(); F.mse_loss(s.dec(s.enc(xb)), xb).backward(); opt.step()
        with torch.no_grad():
            s.recon = float((torch.linalg.norm(s.dec(s.enc(Pub)) - Pub, dim=1) / torch.linalg.norm(Pub, dim=1)).median())
        for p_ in s.parameters(): p_.requires_grad_(False)
    def psi(s, Z): return s.dec(Z.T).T
    def psi_batch(s, Z):
        P, k, N = Z.shape
        return s.dec(Z.permute(0, 2, 1).reshape(-1, k)).reshape(P, N, -1).permute(0, 2, 1)
    def coords(s, X): return s.enc(X.T).T
    def std(s, Pub): return s.coords(Pub[:5000].T).std(dim=1, keepdim=True)
    def describe(s): return f"public autoencoder decoder k={s.k} on {s.n_pub} images (median public reconstruction error {s.recon:.3f})"


CHARTS = {"pca": PCAChartLocal, "ae": AEChartLocal}


def parse_chart_name(name):
    """'pca' / 'ae' -> pooled scope (the earlier rows' chart); 'pca_perclass' / 'ae_perclass' -> per-class scope."""
    kind, scope = (name[:-9], "per_class") if name.endswith("_perclass") else (name, "pooled")
    if kind not in CHARTS: raise ValueError(f"unknown chart '{name}'")
    return kind, scope


class ChartSet:
    """One chart per private slot. pooled: ONE chart fitted on the union of the public pools, shared by every slot (the
    earlier rows' chart). per_class: one chart per added class, fitted on that class's public pool, each slot using
    its own class's chart."""
    def __init__(s, name, Pub_list, cls_of, k, dev, shape, ae_epochs):
        s.name = name; s.kind, s.scope = parse_chart_name(name); s.k = k
        Cls = CHARTS[s.kind]
        if s.scope == "pooled" or len(Pub_list) == 1:
            s.pools = [torch.cat(Pub_list, 0)]; s.cls_of = torch.zeros_like(cls_of)
        else:
            s.pools = list(Pub_list); s.cls_of = cls_of
        s.charts = [Cls(P, k, dev, shape, ae_epochs) for P in s.pools]
        s.slots = [torch.nonzero(s.cls_of == c).flatten() for c in range(len(s.charts))]

    def _per_slot(s, f, Z, dim):
        if len(s.charts) == 1: return f(s.charts[0], Z)
        parts = [(idx, f(s.charts[c], Z.index_select(dim, idx))) for c, idx in enumerate(s.slots) if len(idx)]
        shp = list(parts[0][1].shape); shp[dim] = Z.shape[dim]
        out = torch.zeros(shp, dtype=parts[0][1].dtype, device=Z.device)
        for idx, p in parts: out = out.index_copy(dim, idx, p)
        return out
    def psi(s, Z): return s._per_slot(lambda ch, z: ch.psi(z), Z, 1)                    # (k, N) -> (D, N)
    def psi_batch(s, Z): return s._per_slot(lambda ch, z: ch.psi_batch(z), Z, 2)        # (P, k, N) -> (P, D, N)
    def coords(s, X): return s._per_slot(lambda ch, x: ch.coords(x), X, 1)              # (D, N) -> (k, N)
    def std(s):                                                                          # (k, N): each slot's chart's std on its own pool
        sd = [s.charts[c].std(s.pools[c]) for c in range(len(s.charts))]
        return torch.cat([sd[int(c)] for c in s.cls_of.tolist()], 1)
    def chart_of_start(s, p):
        """Certificate arm: start p runs on chart (p mod n_charts) from the coordinates of that chart's first slot."""
        c = p % len(s.charts); return s.charts[c], int(s.slots[c][0])
    def control(s, X_on):
        """Release-free baseline per slot: the nearest PUBLIC image of the slot's chart pool, on that chart."""
        imgs, errs = [], []
        with torch.no_grad():
            for i in range(X_on.shape[1]):
                c = int(s.cls_of[i]); ch, P = s.charts[c], s.pools[c]
                Pon = torch.cat([ch.psi(ch.coords(P[j:j + 1000].T)) for j in range(0, len(P), 1000)], 1)
                d = torch.linalg.norm(Pon - X_on[:, i:i + 1], dim=0) / torch.linalg.norm(X_on[:, i]); j = int(d.argmin())
                imgs.append(Pon[:, j]); errs.append(float(d[j]))
        return torch.stack(imgs, 1), errs
    def describe(s): return f"{s.name} [{s.scope}, {len(s.charts)} chart(s)]: " + " | ".join(ch.describe() for ch in s.charts)


# ------------------------------------------------------------------------------------------------ per-cell outputs
def save_cell(a, tag, ch, k, T, pn, cell, shape, cname, ncls):
    """Per-cell tensors (.pth) and a PNG grid: truth raw / chart target / NTK best / certificate best / control."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    stem = f"{tag}_{ch}_k{k}_T{T}"
    torch.save(dict(cell=cell, **pn), os.path.join(a.save_dir, f"{stem}.pth"))
    rows_fig = [("private (raw)", pn["x_raw"], None), ("chart projection = the target", pn["x_chart"], None),
                ("NTK linearised, free coefficients", pn["ntk"], pn["ntk_err"]), ("certificate", pn["cert"], pn["cert_err"]),
                ("control: nearest PUBLIC image on chart", pn["control"], pn["control_err"])]
    N = pn["x_raw"].shape[1]; ce = np.asarray(pn["cert_err"]); ib, iw = int(ce.argmin()), int(ce.argmax())
    fig, ax = plt.subplots(len(rows_fig), N, figsize=(1.35 * N + 2.6, 1.6 * len(rows_fig) + 0.9)); fig.subplots_adjust(left=0.21, top=0.86, bottom=0.02, hspace=0.4)
    for ri, (name, imgs, errs) in enumerate(rows_fig):
        for j in range(N):
            im = imgs[:, j].reshape(*shape).permute(1, 2, 0).clamp(0, 1).float().numpy()
            ax[ri, j].imshow(im.squeeze(), cmap=None if shape[0] == 3 else "gray", vmin=None if shape[0] == 3 else 0, vmax=None if shape[0] == 3 else 1)
            ax[ri, j].axis("off")
            if errs is not None:
                e = errs[j]; ax[ri, j].set_title(("landed" if e < 1e-2 else f"err {e:.2f}") + (" (best)" if ri == 3 and j == ib else " (worst)" if ri == 3 and j == iw else ""), fontsize=6.5)
            elif ri == 0 and pn.get("case"):
                ax[ri, j].set_title(f"{pn['cls_name'][j]} {pn['case'][j]}", fontsize=6.5)
        p_ = ax[ri, 0].get_position(); fig.text(0.012, (p_.y0 + p_.y1) / 2, name, fontsize=7.5, va="center")
    fig.suptitle(f"'{cname}' as {'a new class' if ncls == 1 else f'{ncls} new classes'}{' (ONE head row)' if cell['same_row'] else ''} on {cell['backbone']['backbone']} — "
                 f"head LoRA r={cell['r']}, T={T} SGD steps (lr={cell['lr']}), {cell['chart_desc'][:90]}\n"
                 f"NTK linearised, free coefficients, {cell['ntk_main_form']} form, recovered {cell['ntk_images_found']}/{N}; certificate recovered "
                 f"{cell['cert_images_found']}/{N} ({cell['cert_landed_starts']}/{cell['starts']} starts). "
                 f"Model floor at the truth: {cell['model_floor_at_truth'].get(cell['ntk_main_form'].split(':')[0], float('nan')):.2e}", fontsize=8.5)
    fig.savefig(os.path.join(a.fig_dir, f"{stem}.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar", "mnist"], default="cifar")
    ap.add_argument("--backbone", choices=sorted(CIFAR_BACKBONES), default="cnn", help="CIFAR base model (see CIFAR_BACKBONES); mnist ignores it")
    ap.add_argument("--newclass", default="cifar100:keyboard"); ap.add_argument("--newclass2", default=None)
    ap.add_argument("--letter", default="a"); ap.add_argument("--letter2", default=None)
    ap.add_argument("--same-row", action="store_true", help="two classes' images but ONE new head row (mixed content, single label)")
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--charts", nargs="*", default=["pca", "ae"], help="pca, ae (ONE chart on the union of the public pools, as in every earlier row); pca_perclass, ae_perclass (one chart per added class)")
    ap.add_argument("--ks", nargs="*", type=int, default=[16, 32, 48])
    ap.add_argument("--Ts", nargs="*", type=int, default=[1, 400], help="fine-tuning steps: T=1 is where the linearisation is exact by construction, T=400 is the normal fine-tune")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--starts", type=int, default=200); ap.add_argument("--adam_iters", type=int, default=4000); ap.add_argument("--adam_lr", type=float, default=5e-2)
    ap.add_argument("--lm_iters", type=int, default=300); ap.add_argument("--ae_epochs", type=int, default=150)
    ap.add_argument("--ntk-solvers", nargs="*", default=["varpro", "joint"], help="varpro: eliminate the coefficients in closed "
                    "form and search the latents alone (the right algorithm, and the fair one -- the objective becomes the target "
                    "projected off the span of the candidate features, the same shape as the certificate's). joint: Adam over latents "
                    "AND coefficients together, which is what Experiment B does; kept so the handicap is measured rather than assumed, "
                    "since with the coefficients initialised at zero the latents receive exactly zero gradient on the first step.")
    ap.add_argument("--ntk-forms", nargs="*", default=["lora", "dW"], help="lora: fit the released B_T with sum_i r_i (A_T phi_i)^T (the fair form, exact at T=1). dW: fit the merged Delta W with sum_i r_i phi_i^T (Experiment B verbatim, mis-specified for LoRA)")
    ap.add_argument("--oracle-diag", action="store_true", help="DIAGNOSTIC ONLY, off by default: additionally run the linearised route with ORACLE "
                    "coefficients at the step counts in --oracle-diag-Ts. It uses the private images and is therefore an UPPER BOUND, never an attack "
                    "result; it exists to separate two candidate obstructions at T=1, where the linearisation is exact by construction. If the oracle "
                    "arm recovers and the free arm does not, the obstruction is the free coefficients absorbing a recombination (superposition). If "
                    "both fail, superposition is not the whole account.")
    ap.add_argument("--oracle-diag-Ts", nargs="*", type=int, default=[1])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt", default=None, help="override the CIFAR checkpoint path selected by --backbone"); ap.add_argument("--model-mnist", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--data-root", default="data"); ap.add_argument("--mnist-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default="results/ntk_vs_cert"); ap.add_argument("--fig-dir", default="figures/ntk_vs_cert")
    a = ap.parse_args(); dev = torch.device(a.device)
    os.makedirs(a.save_dir, exist_ok=True); os.makedirs(a.fig_dir, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    from experiments.cifar.cifar_newclass import lm_cert

    phi, W0, Pub_list, Pri_list, cname, binfo, shape, bbname = (build_cifar if a.dataset == "cifar" else build_mnist)(a, dev)
    m, n = W0.shape; D = int(np.prod(shape))
    g = torch.Generator().manual_seed(a.seed + 7)
    ncls = len(Pri_list); per = a.N // ncls; n_rows = 1 if a.same_row else ncls
    sel = [torch.randperm(Pri_list[c].shape[0], generator=g)[: per] for c in range(ncls)]
    X_raw = torch.cat([Pri_list[c][sel[c]].T for c in range(ncls)], 1).contiguous()
    cls_of = torch.cat([torch.full((per,), c, device=dev) for c in range(ncls)])
    y = torch.cat([torch.full((per,), m - n_rows + (0 if a.same_row else c), device=dev) for c in range(ncls)])
    cls_names = cname.split("+"); priv_cls = [cls_names[int(c)] for c in cls_of.tolist()]; priv_idx = torch.cat(sel).tolist()
    case = None
    if a.dataset == "mnist":
        case = emnist_case_of(a.mnist_root, X_raw.T.cpu().numpy(), cls_names)
        log(f"# EMNIST case of each private image (exact byte match against the byclass split): " +
            ", ".join(f"{pc}[{pi}]={ci['case']}({ci['byclass_chars'] or '-'})" for pc, pi, ci in zip(priv_cls, priv_idx, case)))
    log(f"# private join keys (index into each class's held-out TEST pool): " + ", ".join(f"{pc}[{pi}]" for pc, pi in zip(priv_cls, priv_idx)))
    log(f"# ntk_vs_certificate  dataset={a.dataset}  base={bbname}  class='{cname}' ({ncls} added class(es), {n_rows} new head row(s){', SAME ROW' if a.same_row else ''}, labels {y.tolist()})  "
        f"base test {binfo['test_acc']*100:.1f}% train {binfo['train_acc']*100:.2f}%  m={m} n={n} N={a.N} r={a.r}  charts={a.charts} ks={a.ks} Ts={a.Ts}  host={socket.gethostname()}")
    Y = torch.eye(m, device=dev, dtype=torch.float64)[y].T
    A0 = (1.0 / math.sqrt(n) * torch.randn(a.r, n, generator=torch.Generator().manual_seed(a.seed + 7), dtype=torch.float64)).to(dev)
    tag = f"{a.dataset}_{bbname}_{cname.replace('+', '_and_')}{'_samerow' if a.same_row else ''}_N{a.N}_r{a.r}"

    cells = []; panels = {}
    for chart_name in a.charts:
        if chart_name.endswith("_perclass") and ncls == 1:
            log(f"\n### chart={chart_name}: skipped, identical to '{chart_name[:-9]}' on a single-class cell"); continue
        for k in a.ks:
            chart = ChartSet(chart_name, Pub_list, cls_of, k, dev, shape, a.ae_epochs)
            coord_std = chart.std()
            X_on = chart.psi(chart.coords(X_raw))
            repr_err = torch.linalg.norm(X_on - X_raw, dim=0) / torch.linalg.norm(X_raw, dim=0)
            X_ctrl, ctrl_err = chart.control(X_on)
            H = phi(X_on)
            Z0 = (torch.randn(a.starts, k, a.N, generator=torch.Generator().manual_seed(a.seed + 31)).to(dev) * coord_std[None])
            log(f"\n### chart={chart.describe()};  RAW-image projection error of the true privates (before projection; the only cross-chart comparison): "
                f"median {float(np.median(repr_err.cpu().numpy())):.3f}, range {float(repr_err.min()):.3f}-{float(repr_err.max()):.3f}, per image {[f'{v:.3f}' for v in repr_err.tolist()]};  "
                f"control (nearest public on chart) per image {[f'{v:.3f}' for v in ctrl_err]}")
            for T in a.Ts:
                # ---- the release, at this T
                A, B = A0.clone(), torch.zeros(m, a.r, dtype=torch.float64, device=dev)
                for t in range(T):
                    z = W0 @ H + B @ (A @ H); Dm = (torch.softmax(z, 0) - Y) / a.N
                    B, A = B - a.lr * (Dm @ (A @ H).T), A - a.lr * (B.T @ Dm @ H.T)
                A_T, B_T = A, B; dW = B_T @ A_T; ndW = torch.linalg.norm(dW)
                sB = torch.linalg.svdvals(B_T); Np = int((sB > 1e-12 * sB[0]).sum())
                _, _, VhB = torch.linalg.svd(B_T, full_matrices=False); Q = VhB[:Np].T
                C = A_T - Q @ (Q.T @ A_T)
                with torch.no_grad():
                    cert_truth = torch.linalg.norm(C @ H, dim=0) / torch.linalg.norm(A_T @ H, dim=0)
                    R_true = torch.softmax(W0 @ H, 0) - Y                       # true per-sample residuals at the PUBLIC base
                # per form: the target, the feature matrix the coefficients multiply, and the model floor at the truth
                FORMS = {"lora": (B_T, A_T @ H), "dW": (dW, H)}
                floors = {}
                for fname in a.ntk_forms:
                    tgt, Feat = FORMS[fname]
                    # min_R ||tgt - R Feat^T||_F  <=>  min_{R^T} ||Feat R^T - tgt^T||_F, so lstsq takes (Feat, tgt^T)
                    R_ls = torch.linalg.lstsq(Feat, tgt.T).solution.T
                    floors[fname] = float(torch.linalg.norm(tgt - R_ls @ Feat.T) / torch.linalg.norm(tgt))
                log(f"  -- T={T}: rank B_T={Np}, rank C={int(torch.linalg.matrix_rank(C, rtol=1e-10))}, ||dW||={ndW:.2e}, ||B_T||={float(torch.linalg.norm(B_T)):.2e}; "
                    f"certificate residual at the truths {float(cert_truth.max()):.1e}; model floor at the truth " +
                    ", ".join(f"{fn}={v:.2e}" for fn, v in floors.items()))

                def err_matrix(Xc):
                    return torch.stack([torch.linalg.norm(Xc - X_on[:, i:i + 1], dim=0) / torch.linalg.norm(X_on[:, i]) for i in range(a.N)], 1)

                # ---- NTK. coef="free" is THE ATTACK; coef="oracle" is a labelled diagnostic (joint solver only).
                def run_ntk(coef, fname, solver):
                    tgt, _ = FORMS[fname]; ntgt = torch.linalg.norm(tgt); tgtT = tgt.T.contiguous()
                    t0 = time.time(); Z = Z0.clone().requires_grad_(True); Rc = None; params = [Z]
                    if solver == "joint":
                        if coef == "free":
                            Rc = torch.zeros(a.starts, m, a.N, device=dev, requires_grad=True); params = [Z, Rc]
                        else:
                            Rc = (-(a.lr / a.N) * R_true)[None].expand(a.starts, m, a.N).contiguous()
                    opt = torch.optim.Adam(params, a.adam_lr)
                    eyeN = torch.eye(a.N, device=dev)

                    def resid(Z_):
                        Xc = chart.psi_batch(Z_)
                        Hc = phi(Xc.permute(1, 0, 2).reshape(D, -1)).reshape(n, a.starts, a.N).permute(1, 0, 2)
                        Fc = torch.einsum("rn,pnN->prN", A_T, Hc) if fname == "lora" else Hc      # (P, c, N)
                        if solver == "joint":
                            pred = torch.einsum("pmN,pcN->pmc", Rc, Fc)
                            r = torch.linalg.norm((pred - tgt[None]).reshape(a.starts, -1), dim=1) / ntgt
                        else:
                            # VARIABLE PROJECTION: the model is linear in the coefficients, so eliminate them exactly.
                            # min_R ||tgt - R F^T|| leaves the target projected off col(F); no coefficients to optimise,
                            # no zero-gradient start, no two-block scale mismatch, and Nk unknowns instead of N(k+m).
                            Ft = Fc.transpose(1, 2)                                               # (P, N, c)
                            G = Ft @ Fc                                                           # (P, N, N)
                            ridge = (1e-12 * torch.diagonal(G, dim1=1, dim2=2).sum(1) / a.N).clamp(min=1e-300)
                            Rsol = torch.linalg.solve(G + ridge[:, None, None] * eyeN, Ft @ tgtT)  # (P, N, m)
                            r = torch.linalg.norm((Fc @ Rsol - tgtT[None]).reshape(a.starts, -1), dim=1) / ntgt
                        return Xc, r

                    for it in range(a.adam_iters):
                        _, r = resid(Z)
                        opt.zero_grad(); r.sum().backward(); opt.step()
                    with torch.no_grad():
                        Xc, r = resid(Z)
                        cand_ = Xc.detach().permute(1, 0, 2).reshape(D, -1)
                        E_ = err_matrix(cand_); best_ = E_.min(0).values
                    return dict(res=r.detach(), best=best_, idx=E_.argmin(0), found=int((best_ < 1e-2).sum()),
                                sec=time.time() - t0, cand=cand_, solver=solver, form=fname, coef=coef)

                ntk_by_form = {}
                for fname in a.ntk_forms:
                    for solver in a.ntk_solvers:
                        nt = run_ntk("free", fname, solver); ntk_by_form[f"{fname}:{solver}"] = nt
                        log(f"     NTK[{fname:4s}/{solver:6s}] free coefficients: residual {float(nt['res'].min()):.3e} "
                            f"(median {float(nt['res'].median()):.3e}; model floor at the truth {floors[fname]:.2e}); "
                            f"images recovered {nt['found']}/{a.N}; closest per image {[f'{v:.3f}' for v in nt['best'].tolist()]}  [{nt['sec']:.0f}s]")
                    if a.oracle_diag and T in a.oracle_diag_Ts and "joint" in a.ntk_solvers:
                        oc = run_ntk("oracle", fname, "joint"); ntk_by_form[f"{fname}:joint_ORACLE_DIAG"] = oc
                        log(f"     NTK[{fname:4s}/joint ] ORACLE coefficients — DIAGNOSTIC UPPER BOUND, uses the private images, NOT an attack: "
                            f"residual {float(oc['res'].min()):.3e}; images recovered {oc['found']}/{a.N}  [{oc['sec']:.0f}s]")
                main_form = ("lora:varpro" if "lora:varpro" in ntk_by_form else list(ntk_by_form)[0])
                ntk = ntk_by_form[main_form]
                ntk_res, ntk_best, ntk_idx, ntk_found, ntk_sec, cand = ntk["res"], ntk["best"], ntk["idx"], ntk["found"], ntk["sec"], ntk["cand"]

                # ---- certificate, same starts, native LM; start p runs on chart (p mod n_charts)
                t0 = time.time(); Xs, objs = [], []
                for p in range(a.starts):
                    ch_p, slot_p = chart.chart_of_start(p)
                    fun = lambda w, ch_=ch_p: (C @ phi(ch_.psi(w.reshape(k, 1)))).reshape(-1) / torch.linalg.norm(A_T @ phi(ch_.psi(w.reshape(k, 1))))
                    w, obj, _ = lm_cert(fun, Z0[p, :, slot_p].clone(), a.lm_iters); Xs.append(ch_p.psi(w.reshape(k, 1))[:, 0]); objs.append(obj)
                Xs = torch.stack(Xs, 1); objs = torch.tensor(objs, device=dev)
                with torch.no_grad():
                    E = err_matrix(Xs); cert_best = E.min(0).values; cert_idx = E.argmin(0)
                    cert_found = int((cert_best < 1e-2).sum()); landed = int((E.min(1).values < 1e-2).sum())
                    order = torch.argsort(objs); top20 = sum(int(E[j].min() < 1e-2) for j in order[:20].tolist())
                cert_sec = time.time() - t0
                log(f"     CERT (same starts, LM)  : residual {float(objs.min())**0.5:.3e}; images recovered {cert_found}/{a.N}; "
                    f"landed starts {landed}/{a.starts}; top-20 by residual landed {top20}/20; closest per image {[f'{v:.3f}' for v in cert_best.tolist()]}  [{cert_sec:.0f}s]")

                cell = dict(part="ntk_vs_cert", dataset=a.dataset, class_name=cname, n_classes=ncls, same_row=a.same_row, head_rows=n_rows, labels=y.tolist(),
                            private_class=priv_cls, private_idx=priv_idx, private_case=([c_["case"] for c_ in case] if case else None),
                            private_case_detail=case, chart=chart_name, chart_kind=chart.kind, chart_scope=chart.scope, chart_desc=chart.describe(),
                            chart_slot_assignment=("each slot uses its own class's chart (attacker told the slot classes)" if len(chart.charts) > 1 else "one chart for every slot"),
                            k=k, T=T, lr=a.lr, N=a.N, r=a.r, m=m, n=n, seed=a.seed, starts=a.starts, n_prime=Np,
                            chart_repr_err_median=float(np.median(repr_err.cpu().numpy())), chart_repr_err=repr_err.tolist(),
                            control_public_nn_err=ctrl_err, control_def="nearest public image of the slot's chart pool, projected on that chart, same relative-error metric",
                            cert_residual_at_truth=cert_truth.tolist(), model_floor_at_truth=floors, ntk_main_form=main_form,
                            ntk_verdict={fn: ("alias: residual AT the model floor, wrong images (identifiability)" if v["found"] < a.N and float(v["res"].min()) <= 3 * max(floors.get(fn.split(":")[0], 0.0), 1e-14)
                                              else "search failure: residual ABOVE the model floor" if v["found"] < a.N else "recovered")
                                         for fn, v in ntk_by_form.items()},
                            ntk_by_form={fn: dict(images_found=v["found"], residual_min=float(v["res"].min()),
                                                  residual_median=float(v["res"].median()), closest_per_image=v["best"].tolist(),
                                                  model_floor_at_truth=floors.get(fn.split(":")[0]), solver=v["solver"],
                                                  is_diagnostic_upper_bound=fn.endswith("_ORACLE_DIAG"))
                                         for fn, v in ntk_by_form.items()},
                            ntk_residual_min=float(ntk_res.min()), ntk_residual_median=float(ntk_res.median()),
                            ntk_images_found=ntk_found, ntk_closest_per_image=ntk_best.tolist(), ntk_seconds=ntk_sec,
                            cert_residual_min=float(objs.min()) ** 0.5, cert_images_found=cert_found, cert_landed_starts=landed,
                            cert_starts_per_chart=[int(sum(1 for p in range(a.starts) if p % len(chart.charts) == c)) for c in range(len(chart.charts))],
                            cert_top20_landed=top20, cert_closest_per_image=cert_best.tolist(), cert_seconds=cert_sec,
                            oracle_note="entries whose key ends in _ORACLE_DIAG are UPPER BOUNDS, NOT attack results: their coefficients "
                                        "are fixed at the T=1 closed form, which uses the private images. They exist only to separate two "
                                        "candidate obstructions: recovering there while the free arm fails means the free coefficients are "
                                        "absorbing a recombination; failing in both means that is not the whole account.",
                            backbone=binfo, host=socket.gethostname(), cmd=" ".join(sys.argv))
                cells.append(cell)
                print(json.dumps(cell), flush=True)
                if a.out:
                    with open(a.out, "a") as f: f.write(json.dumps(cell) + "\n")
                pn = dict(x_raw=X_raw.cpu(), x_chart=X_on.cpu(), ntk=cand[:, ntk_idx].cpu(), cert=Xs[:, cert_idx].cpu(), control=X_ctrl.cpu(),
                          ntk_err=ntk_best.tolist(), cert_err=cert_best.tolist(), control_err=ctrl_err, cls_name=priv_cls, private_idx=priv_idx,
                          case=([c_["case"] for c_ in case] if case else None), y=y.cpu())
                panels[(chart_name, k, T)] = pn
                save_cell(a, tag, chart_name, k, T, pn, cell, shape, cname, ncls)

    # ---------------------------------------------------------------- summary table + figures (aggregate named by the Ts of THIS job)
    agg = f"{tag}_{'-'.join(a.charts)}_Ts{'-'.join(str(T) for T in a.Ts)}"
    torch.save(dict(cells=cells, panels=panels, y=y.cpu()), os.path.join(a.save_dir, f"{agg}.pth"))
    tab = ["RECOVERY (compare only within a chart: each chart defines its own on-chart privates and release)", "",
           "| chart | scope | k | T | raw proj. err (median) | control NN err (median) | model floor at truth | NTK images (lora/varpro) | NTK residual | verdict | certificate images | certificate landed | top-20 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        fl = c["model_floor_at_truth"].get(c["ntk_main_form"].split(":")[0], float("nan"))
        tab.append(f"| {c['chart']} | {c['chart_scope']} | {c['k']} | {c['T']} | {c['chart_repr_err_median']:.3f} | {float(np.median(c['control_public_nn_err'])):.3f} | {fl:.2e} | "
                   f"**{c['ntk_images_found']}/{c['N']}** | {c['ntk_residual_min']:.2e} | {c['ntk_verdict'].get(c['ntk_main_form'], '')[:34]} | "
                   f"**{c['cert_images_found']}/{c['N']}** | {c['cert_landed_starts']}/{c['starts']} | {c['cert_top20_landed']}/20 |")
    fid = ["", "RAW-image projection error per chart and k (the true held-out test images BEFORE projection). Privates are on-chart, so each chart",
           "defines its own privates and release: recovery counts above are comparable only WITHIN a chart; THIS table is the only cross-chart comparison.",
           "(ctrl = nearest public image on the same chart, at the same metric.)", "",
           "| chart | scope | k | median | range | " + " | ".join(f"{pc}[{pi}]" for pc, pi in zip(priv_cls, priv_idx)) + " |", "|---|---|---|---|---|" + "---|" * a.N]
    seen = set()
    for c in cells:
        if (c["chart"], c["k"]) in seen: continue
        seen.add((c["chart"], c["k"]))
        re_ = c["chart_repr_err"]
        fid.append(f"| {c['chart']} | {c['chart_scope']} | {c['k']} | {c['chart_repr_err_median']:.3f} | {min(re_):.3f}-{max(re_):.3f} | " +
                   " | ".join(f"{e:.3f} (ctrl {ce:.3f})" for e, ce in zip(re_, c["control_public_nn_err"])) + " |")
    open(os.path.join(a.fig_dir, f"{agg}_table.md"), "w").write("\n".join(tab + fid) + "\n")
    log("\n=== HEAD TO HEAD (free coefficients only; no oracle anywhere) ===\n" + "\n".join(tab + fid))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # (a) summary: images recovered vs k, per chart type and T
    chart_names = [c for c in a.charts if any(x["chart"] == c for x in cells)]
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, len(a.Ts), figsize=(5.2 * len(a.Ts), 4.2), squeeze=False)
    for ti, T in enumerate(a.Ts):
        ax = axes[0][ti]
        for ci, ch in enumerate(chart_names):
            sel_ = [c for c in cells if c["chart"] == ch and c["T"] == T]
            if not sel_: continue
            ks = [c["k"] for c in sel_]
            ax.plot(ks, [c["cert_images_found"] for c in sel_], "o-", color=cols[ci % len(cols)], label=f"certificate, {ch}")
            ax.plot(ks, [c["ntk_images_found"] for c in sel_], "s--", color=cols[ci % len(cols)], alpha=0.6, label=f"NTK free-c, {ch}")
        ax.set_xlabel("chart dimension k"); ax.set_ylabel(f"private images recovered (of {a.N})"); ax.set_ylim(-0.4, a.N + 0.4)
        ax.set_title(f"T = {T} fine-tuning steps (lr={a.lr})", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Same release, two attacks — '{cname}' as {'a new class' if ncls == 1 else f'{ncls} new classes'}{' (one head row)' if a.same_row else ''} on "
                 f"{bbname}, head LoRA r={a.r}, N={a.N}, {a.starts} shared random starts", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(a.fig_dir, f"{agg}_summary.png"), dpi=200); plt.close(fig)
    log(f"saved {a.save_dir}/{agg}.pth (+ per-cell {tag}_<chart>_k<k>_T<T>.pth) and {a.fig_dir}/{agg}_summary.png, {agg}_table.md, per-cell {tag}_<chart>_k<k>_T<T>.png")


if __name__ == "__main__":
    main()
