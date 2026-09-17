#!/usr/bin/env python3
"""Bootstrap / iterative chart (plan 2026-09-18 WP4; round2 TEST 8), first run. Two variants, two releases.

Question. Can the chart be improved from what the attack recovers, WITHOUT private data touching the chart fit?

Releases (privates RAW, i.e. the release is fine-tuned on the photographs / letters themselves, not on chart points):
  mnist   the strong MNIST MLP (`models/exact_inversion/mnist_mlp_strong.pth`, 784-1000-1000-10 GELU), EMNIST letter
          `a` as an 11th class -- the SAME eight test-split letters as the letters cell (`randperm(seed+7)[:N]`),
          the same A0 (`randn(r, n, seed+7) / sqrt(n)`), head extended by a zero row, r = 64, T = 400, lr = 0.01.
  cifar   the CIFAR-10 CNN (`models/exact_inversion/cifar10_cnn_newclass.pth`, penultimate 256-dim), CIFAR-100
          `motorcycle` as an 11th class -- the ladder's eight test-split photographs, same A0 construction.
Release construction: `train_release(..., "sgd")` (the loop the ladder inlines, verbatim); certificate
C = P_{row(B_T)^perp} A_T from `certificate.certificate`; search = `certificate.lm_cert` (Levenberg-Marquardt on
w -> C phi(psi(w)) / ||A_T phi(psi(w))||, the ladder's `cifar_newclass.lm_cert` is a verbatim copy). Nothing in the
solver is rewritten here; only the CHART changes between rounds.

Round 0 (both variants): a GENERIC public chart the attacker can build without knowing the added class -- PCA-k of
ALL EMNIST letters (train split, 26 classes) / ALL CIFAR-100 train images (100 classes). `--starts` random starts at
the public coordinate scale. The attacker's candidates = the N best-objective non-degenerate starts after greedy
de-duplication; evaluation matches each truth to its nearest candidate (attacker-side selection) and ALSO records
the best start of any kind per truth (oracle-side selection), so the two are never confused.

Variant A -- class-recognition bootstrap. A public classifier trained IN THIS JOB on the public pool only (MNIST:
784-512-26 MLP on EMNIST letters train; CIFAR: the repo's CNN with a 100-way head on CIFAR-100 train; both with a
held-out validation tenth, and both trained with 50 % "chart-projected" augmentation because what they will be
shown are chart points) recognises the class of each round-0 candidate. The class decision for the round is the
argmax of the summed class probabilities over the candidates; the true class's rank in that ordering is recorded,
and so are per-candidate top-1 / top-2. Round 1 chart = PCA-k of the RECOGNISED class's public pool (the public
pool plays the part of "fetch bike images online"). Controls in the same job, same starts: the SECOND-ranked
class's chart (wrong-class refit) and the TRUE class's chart (oracle label, not attacker-available). Round 2 runs
only if re-recognition on the round-1 candidates changes the class (`--a-rounds`).

Variant B -- local-neighbour bootstrap. Per candidate slot j and round t+1: anchor = mean of the K nearest public
images (of the recognised class) to x_j^(t); directions = top-k PCA of those neighbours; attack from x_j^(t)
(warm start, projected into the new chart) and from `--local-starts` fresh random starts; the new candidate is the
best-objective non-degenerate result. Controls: RANDOM-ANCHOR (neighbours of a random public image of the same class
instead of the candidate; the decisive control) and ORACLE-ANCHOR (neighbours of the truth itself).

Per round, per image (audit 2026-09-18): chart error of the TRUE image in that round's chart; TWO recovery errors --
`err_proj`, recovery-to-PROJECTION (the truth projected into that round's chart: did the solver reach the chart's
best) and `err`, recovery-to-truth (did the chart improve); the certificate objective at the recovery; the per-round
FLOOR = the objective at the truth's projection into that round's chart; flags `reached_projection` (err_proj <
1e-2), `landed` (err < 1e-2 vs the raw truth -- dead by construction while the chart error is ~0.3, reported anyway),
`at_floor` (objective at or below the floor), `alias_flag` (at the floor but NOT at the projection: an equally good
chart point that is a different image -- an information problem), `solver_short` (above the floor and not at the
projection -- an optimisation problem); recognised class + rank. Void (audit): round-0 median err_proj > 1e-2 means
the solver did not reach the chart -> `void_round0`; variant A is void unless >= 5 of 8 round-0 candidates are
top-1 correct AND the classifier's calibration accuracy on PCA-k projections of held-out public images of the true
class is above that 5/8 bar -> `void_variant_A`. The run still completes so the numbers exist; the verdict is VOID,
not null.

Reference change (coordinator 2026-09-18, after smoke 355880): the projection is not the chart's argmin of the objective
for raw privates, so per round and per image the same solver is ALSO run from the ORACLE start (coords of the truth's
projection; not attacker-available) to give x*_chart. Columns `err_opt` = err(recovery, x*_chart) [solver metric],
`err_opt_truth` = err(x*_chart, truth) [chart fidelity], `objective` vs `objective_opt`; flags `reached_opt`,
`alias_in_chart` (objective <= the oracle-start optimum but far from it: an alias within the chart, not void),
`solver_short_opt`. Void round 0 = median err_opt > 1e-2 AND objective above the oracle-start optimum on > N/2 images.

Base-model gate (WP0): train accuracy, test accuracy and train loss of the frozen backbone are MEASURED in-job at
load time (not copied from a note) and stamped on every row.

  python -u -m experiments.bootstrap_chart.bootstrap --releases mnist --starts 10 --a-rounds 1 --b-rounds 1   # smoke
  python -u -m experiments.bootstrap_chart.bootstrap --releases mnist cifar                                     # full
"""
import argparse, json, math, os, pickle, socket, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from experiments.exact_inversion.lora_exact_inversion import train_release, git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, read_idx
from experiments.exact_inversion.certificate import certificate, lm_cert
from experiments.cifar.cifar_newclass import train_backbone, load_cifar100_class, load_cifar10, CNN

torch.set_default_dtype(torch.float64)
FLOOR = 1e-20          # certificate.py's `at_floor` threshold on the objective ||C phi||^2 / ||A_T phi||^2
LAND = 1e-2            # relative image error below which a recovery counts as landed (ladder / RECOVER_TOL)
DEDUPE = 0.05          # relative distance below which two chart images are the same candidate
GATE = (0.995, 1e-2)   # WP0: train acc >= 99.5 % and train loss <= 1e-2


def log(s): print(s, flush=True)


# ------------------------------------------------------------------------------------------------- data
def load_emnist_all(root):
    """EMNIST 'letters' split, ALL 26 classes: images transposed to MNIST orientation, [0, 1], labels 0..25.
       Same parsing as new_class.load_emnist_letters, so `Xte[yte == 0]` is that loader's letter-a test pool in the
       same order (the private selection `randperm(seed+7)[:N]` then picks the letters cell's eight)."""
    out = {}
    for split in ("train", "test"):
        d = os.path.join(root, "EMNIST", "raw")
        with open(os.path.join(d, f"emnist-letters-{split}-images-idx3-ubyte"), "rb") as f:
            f.read(16); img = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)
        with open(os.path.join(d, f"emnist-letters-{split}-labels-idx1-ubyte"), "rb") as f:
            f.read(8); lab = np.frombuffer(f.read(), dtype=np.uint8)
        out[split] = (img.astype(np.float64) / 255.0, lab.astype(np.int64) - 1)
    return out


def load_cifar100_all(root):
    """All CIFAR-100 train images with fine labels (float32/255 -> float64, the ladder's numeric path) and the names."""
    meta = pickle.load(open(os.path.join(root, "cifar-100-python", "meta"), "rb"), encoding="bytes")
    names = [n.decode() for n in meta[b"fine_label_names"]]
    b = pickle.load(open(os.path.join(root, "cifar-100-python", "train"), "rb"), encoding="bytes")
    return b[b"data"].astype(np.float32) / 255.0, np.array(b[b"fine_labels"], dtype=np.int64), names


# ------------------------------------------------------------------------------------------------- charts
class Chart:
    """psi(w) = mean + V w, coords(x) = V^T (x - mean): the same affine chart as PCAChart / ladder_cell, built from any
       pool through the covariance eigendecomposition (identical V to their SVD; fits a 124,800-image pool)."""
    def __init__(s, mean, V, name, explained=float("nan"), pool_size=0):
        s.mean, s.V, s.k, s.name, s.explained, s.pool_size = mean, V, V.shape[1], name, explained, pool_size

    @staticmethod
    def pca(pool, k, name):
        mean = pool.mean(0); Xc = pool - mean
        ev, U = torch.linalg.eigh(Xc.T @ Xc)
        V = U[:, -k:].flip(1).contiguous()
        return Chart(mean, V, name, float(ev[-k:].sum() / ev.sum().clamp_min(1e-300)), pool.shape[0])

    def psi(s, W): return s.mean[:, None] + s.V @ W
    def coords(s, X): return s.V.T @ (X - s.mean[:, None])
    def project(s, X): return s.psi(s.coords(X))
    def err(s, X): return torch.linalg.norm(s.project(X) - X, dim=0) / torch.linalg.norm(X, dim=0)
    def std(s, pool): return s.coords(pool[:5000].T).std(dim=1, keepdim=True)
    def same_as(s, o): return s.V.shape == o.V.shape and bool((s.mean - o.mean).abs().max() < 1e-12 and (s.V - o.V).abs().max() < 1e-12)


def local_chart(anchor_x, pool, K, k, name):
    """The 'local families' chart: K nearest public images (L2) to anchor_x, mean + top-k PCA of those neighbours."""
    d = torch.linalg.norm(pool - anchor_x[None, :], dim=1)
    nb = pool[d.topk(min(K, pool.shape[0]), largest=False).indices]
    return Chart.pca(nb, k, name), nb


# ------------------------------------------------------------------------------------------------- release
class Release:
    """The released factors, the certificate, and the objective -- built once per release, never refit."""
    def __init__(s, name, phi, W0, X_raw, y, A0, T, lr, tol, feat_pool):
        s.name, s.phi, s.X_raw, s.y, s.N = name, phi, X_raw, y, X_raw.shape[1]
        s.m, s.n, s.r = W0.shape[0], W0.shape[1], A0.shape[0]
        H = phi(X_raw)
        s.A_T, s.B_T = train_release(H, A0, W0, y, s.m, T, lr, "sgd")
        s.C, s.Np, s.sB = certificate(s.A_T, s.B_T, tol)
        s.rank_C = int(torch.linalg.matrix_rank(s.C, rtol=1e-10))
        with torch.no_grad():
            s.res_truth = s.obj_of(X_raw)                                            # the certificate at the RAW truths (~1e-16)
            s.feat_ref = float(torch.linalg.norm(s.A_T @ phi(feat_pool[:256].T), dim=0).median())

    def obj_of(s, X):
        f = s.phi(X); return (torch.linalg.norm(s.C @ f, dim=0) / torch.linalg.norm(s.A_T @ f, dim=0)) ** 2

    def fun(s, chart):
        def fun_(w):
            f = s.phi(chart.psi(w.reshape(chart.k, 1)))
            return (s.C @ f).reshape(-1) / torch.linalg.norm(s.A_T @ f)
        return fun_

    def search(s, chart, W0s, iters):
        """LM from each start in W0s (list of (k,) tensors); returns one dict per start."""
        fun = s.fun(chart); out = []
        for w0 in W0s:
            w, obj, it = lm_cert(fun, w0, iters)
            with torch.no_grad():
                x = chart.psi(w.reshape(chart.k, 1))[:, 0]
                fr = float(torch.linalg.norm(s.A_T @ s.phi(x.reshape(-1, 1))) / s.feat_ref)
            out.append(dict(w=w.detach(), x=x, objective=obj, iters=it, feat_ratio=fr, degenerate=bool(fr < 0.05)))
        return out


def rel_err(x, X): return (torch.linalg.norm(X - x[:, None], dim=0) / torch.linalg.norm(X, dim=0))


def pick_candidates(res, N):
    """The attacker's own selection: best objective first, non-degenerate, greedily de-duplicated, at most N."""
    order = sorted([i for i, r in enumerate(res) if not r["degenerate"]], key=lambda i: res[i]["objective"]) or \
            sorted(range(len(res)), key=lambda i: res[i]["objective"])
    keep = []
    for i in order:
        if all(float(rel_err(res[i]["x"], res[j]["x"][:, None])[0]) > DEDUPE for j in keep): keep.append(i)
        if len(keep) == N: break
    return keep


def oracle_optimum(rel, chart, X_targets, iters):
    """NOT ATTACKER-AVAILABLE. The chart optimum nearest each target: the same LM solver started at w0 = coords of the
       target's projection into the chart (coordinator 2026-09-18: the projection itself is not the chart's argmin of the
       certificate objective when the privates are raw, so the reference for 'did the solver reach the chart's best' is
       x*_chart, and err(x*_chart, truth) is the chart's fidelity -- the number a better chart must reduce)."""
    res = rel.search(chart, [w for w in chart.coords(X_targets).T], iters)
    X_opt = torch.stack([r["x"] for r in res], 1)
    return dict(X=X_opt, objective=[r["objective"] for r in res], iters=[r["iters"] for r in res],
                err_truth=[float(v) for v in torch.linalg.norm(X_opt - X_targets, dim=0) / torch.linalg.norm(X_targets, dim=0)],
                objective_at_projection=[float(v) for v in rel.obj_of(chart.project(X_targets))])


def flags(err, err_proj, obj, floor, err_opt=None, obj_opt=None):
    """Projection-based columns (audit 2026-09-18): the floor is the objective at the truth's projection into THIS chart;
       'reached_projection' against the projection, 'landed' against the raw truth; at_floor means at-or-below.
       Optimum-based columns (coordinator 2026-09-18): against x*_chart from the oracle start -- 'reached_opt' (solver metric),
       'alias_in_chart' (objective at or below the oracle-start optimum but far from it: a different chart point that scores
       as well -- an information statement, not a solver failure), 'solver_short_opt' (above it and far from it)."""
    at_floor = bool(obj <= floor * (1 + 1e-6) + FLOOR); reached = bool(err_proj < LAND); landed = bool(err < LAND)
    d = dict(landed=landed, reached_projection=reached, at_floor=at_floor, objective_over_floor=float(obj / max(floor, FLOOR)),
             alias_flag=bool(at_floor and not reached), solver_short=bool(not at_floor and not reached), unverified_landing=bool(landed and not at_floor))
    if err_opt is None:
        d.update(opt_available=False, reached_opt=None, alias_in_chart=None, solver_short_opt=None, obj_above_opt=None, objective_over_opt=None)
    else:
        above = bool(obj > obj_opt * (1 + 1e-6) + FLOOR); r_opt = bool(err_opt < LAND)
        d.update(opt_available=True, reached_opt=r_opt, alias_in_chart=bool(not r_opt and not above), solver_short_opt=bool(not r_opt and above),
                 obj_above_opt=above, objective_over_opt=float(obj / max(obj_opt, FLOOR)))
    return d


# ------------------------------------------------------------------------------------------------- recognition
class LetterMLP(nn.Module):
    def __init__(s, ncls=26):
        super().__init__(); s.net = nn.Sequential(nn.Linear(784, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, ncls))
    def forward(s, x): return s.net(x.reshape(len(x), -1))


def train_classifier(domain, Xpub, ypub, ncls, epochs, chart, dev, cache, seed):
    """Public classifier on the public pool only (a validation tenth held out for the accuracy number). Half of every
       batch is passed through the public chart (projection) because the images it will be asked about are chart
       points. Returns the net and its validation accuracy on raw and on chart-projected validation images."""
    net = (LetterMLP(ncls) if domain == "mnist" else CNN(m=ncls)).to(dev).float()      # FP32 on purpose (module default is FP64)
    g = torch.Generator().manual_seed(seed + 5); perm = torch.randperm(len(Xpub), generator=g)
    nval = len(Xpub) // 10; iv, it = perm[:nval].to(dev), perm[nval:].to(dev)
    X32 = Xpub.float(); y = ypub
    proj = lambda xb: chart.project(xb.double().T).T.float()

    def calibrate():
        """Accuracy on the held-out tenth, raw and PCA-k-PROJECTED through the round-0 chart, overall and PER CLASS (the
           calibration the audit asks for: what recognition on a chart point of this class can be expected at all)."""
        net.eval()
        with torch.no_grad():
            pr_raw = torch.cat([net(X32[i]).argmax(1) for i in iv.split(1000)]); pr_proj = torch.cat([net(proj(X32[i])).argmax(1) for i in iv.split(1000)])
        yv = y[iv]
        per_cls = [float((pr_proj[yv == c] == c).float().mean()) if (yv == c).any() else float("nan") for c in range(ncls)]
        return dict(val_acc_raw=float((pr_raw == yv).float().mean()), val_acc_proj=float((pr_proj == yv).float().mean()), val_acc_proj_per_class=per_cls, n_val=int(nval))

    if os.path.exists(cache):
        blob = torch.load(cache, map_location="cpu", weights_only=False); net.load_state_dict(blob["state_dict"]); net.eval()
        cal = calibrate(); log(f"# classifier loaded from {cache}: val acc raw {cal['val_acc_raw']*100:.1f}%, projected {cal['val_acc_proj']*100:.1f}%")
        return net, cal
    steps = math.ceil(len(it) / 256)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.05, epochs=epochs, steps_per_epoch=steps)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        for idx in it[torch.randperm(len(it), device=dev)].split(256):
            xb, yb = X32[idx], y[idx]
            half = len(xb) // 2
            if half: xb = torch.cat([proj(xb[:half]), xb[half:]], 0)
            if domain == "cifar":
                xb = xb.reshape(-1, 3, 32, 32)
                if torch.rand(1).item() < 0.5: xb = xb.flip(3)
                p = F.pad(xb, (4, 4, 4, 4), mode="reflect"); dx, dy = torch.randint(0, 9, (2,)); xb = p[:, :, dy:dy + 32, dx:dx + 32]
            opt.zero_grad(); F.cross_entropy(net(xb.reshape(len(xb), -1)), yb).backward(); opt.step(); sched.step()
        cal = calibrate()
        log(f"#   classifier epoch {ep+1}/{epochs}: val acc raw {cal['val_acc_raw']*100:.1f}%, chart-projected {cal['val_acc_proj']*100:.1f}%  ({time.time()-t0:.0f}s)")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(dict(state_dict=net.state_dict(), epochs=epochs, domain=domain, ncls=ncls, **cal), cache)
    return net, cal


def recognise(clf, X):
    """X (D, P) float64 -> per-column probabilities (P, ncls) in FP32."""
    with torch.no_grad(): return torch.softmax(clf(X.T.float()), 1)


# ------------------------------------------------------------------------------------------------- figures
def grid(rows, shape, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ncol = max(X.shape[1] for _, X in rows)
    fig, ax = plt.subplots(len(rows), ncol, figsize=(1.3 * ncol + 1.6, 1.35 * len(rows) + 0.8), squeeze=False)
    fig.subplots_adjust(left=0.17, top=0.86, bottom=0.02, hspace=0.25, wspace=0.05)
    for ri, (name, X) in enumerate(rows):
        for c in range(ncol):
            a_ = ax[ri, c]; a_.axis("off")
            if c < X.shape[1]:
                im = X[:, c].detach().cpu().clamp(0, 1)
                if shape[0] == 3: a_.imshow(im.reshape(3, 32, 32).permute(1, 2, 0).numpy())
                else: a_.imshow(im.reshape(28, 28).numpy(), cmap="gray", vmin=0, vmax=1)
        p_ = ax[ri, 0].get_position(); fig.text(0.01, (p_.y0 + p_.y1) / 2, name, fontsize=7, va="center")
    fig.suptitle(title, fontsize=8); fig.savefig(path, dpi=140); plt.close(fig)
    log(f"# figure {path}")


# ------------------------------------------------------------------------------------------------- setup
def base_gate_mnist(bb, root, dev):
    Xtr, ytr = read_idx(root, "train"); Xte, yte = read_idx(root, "test")
    def run(X, y_):
        X = torch.tensor(X, device=dev); y_ = torch.tensor(y_, device=dev); zs = []
        with torch.no_grad():
            for i in range(0, len(X), 5000): zs.append(bb.logits(X[i:i + 5000].T).T)
        z = torch.cat(zs); return float((z.argmax(1) == y_).double().mean()), float(F.cross_entropy(z, y_))
    tr, trl = run(Xtr, ytr); te, _ = run(Xte, yte)
    return dict(train_acc=tr, train_loss=trl, test_acc=te, n_train=len(Xtr), fully_trained=bool(tr >= GATE[0] and trl <= GATE[1]), measured_in_job=True)


def base_gate_cifar(net, root, dev):
    Xtr, ytr, Xte, yte = load_cifar10(root)
    def run(X, y_):
        X = torch.tensor(X, dtype=torch.float64, device=dev); y_ = torch.tensor(y_, device=dev); zs = []
        with torch.no_grad():
            for i in range(0, len(X), 1000): zs.append(net(X[i:i + 1000]))
        z = torch.cat(zs); return float((z.argmax(1) == y_).double().mean()), float(F.cross_entropy(z, y_))
    tr, trl = run(Xtr, ytr); te, _ = run(Xte, yte)
    return dict(train_acc=tr, train_loss=trl, test_acc=te, n_train=len(Xtr), fully_trained=bool(tr >= GATE[0] and trl <= GATE[1]), measured_in_job=True)


def setup(domain, a, dev):
    """Backbone, public pools (with class labels), private truths, and the release. Everything class-related is
       returned as a dict so the two domains share one round loop."""
    if domain == "mnist":
        bb = TrainedBackbone(a.mnist_model, dev, "gelu"); gate = base_gate_mnist(bb, a.mnist_root, dev)
        em = load_emnist_all(a.mnist_root)
        Xpub = torch.tensor(em["train"][0], device=dev); ypub = torch.tensor(em["train"][1], device=dev)
        names = [chr(ord("a") + i) for i in range(26)]; true_cls = names.index(a.letter)
        Pri = torch.tensor(em["test"][0][em["test"][1] == true_cls], device=dev)
        phi = bb.phi; W0 = torch.cat([bb.W0, torch.zeros(1, bb.n, device=dev)], 0); n = bb.n; shape = (1, 28, 28); ckpt = a.mnist_model
    else:
        net, te_, tr_ = train_backbone(a.cifar_ckpt, a.data_root, dev, 40, 0.80, 0.90, "cnn")
        net = net.double()
        for p_ in net.parameters(): p_.requires_grad_(False)
        gate = base_gate_cifar(net, a.data_root, dev)
        Xp, yp, names = load_cifar100_all(a.data_root)
        Xpub = torch.tensor(Xp, dtype=torch.float64, device=dev); ypub = torch.tensor(yp, device=dev)
        pool, cname = load_cifar100_class(a.data_root, a.cifar_class); true_cls = names.index(cname)
        Pri = torch.tensor(pool["test"], dtype=torch.float64, device=dev)                  # the ladder's private pool, same numeric path
        n = net.head.weight.shape[1]; W0 = torch.cat([net.head.weight.double(), torch.zeros(1, n, dtype=torch.float64, device=dev)], 0)
        phi = lambda X: net.phi(X.T).T; shape = (3, 32, 32); ckpt = a.cifar_ckpt
    g = torch.Generator().manual_seed(a.seed + 7); perm = torch.randperm(Pri.shape[0], generator=g)
    X_raw = Pri[perm[: a.N]].T.contiguous()                                                # the SAME privates as the letters cell / ladder
    y = torch.full((a.N,), W0.shape[0] - 1, device=dev)
    A0 = (1.0 / math.sqrt(n) * torch.randn(a.r, n, generator=torch.Generator().manual_seed(a.seed + 7), dtype=torch.float64)).to(dev)
    rel = Release(domain, phi, W0, X_raw, y, A0, a.T, a.lr, a.tol, Xpub)
    log(f"# [{domain}] backbone {ckpt}: MEASURED train {gate['train_acc']*100:.2f}% / test {gate['test_acc']*100:.2f}% / train CE {gate['train_loss']:.2e} "
        f"-> WP0 gate {'PASS' if gate['fully_trained'] else 'FAIL'};  public pool {Xpub.shape[0]} x {Xpub.shape[1]} in {len(names)} classes; "
        f"true class '{names[true_cls]}' (private pool {Pri.shape[0]}, N={a.N} RAW privates idx {perm[:a.N].tolist()})")
    log(f"# [{domain}] release: m={rel.m} n={rel.n} r={a.r} T={a.T} lr={a.lr}; rank B_T = {rel.Np}, rank C = {rel.rank_C}, certificate line k < {a.r - rel.Np} "
        f"(k={a.k}: {'below' if a.k < a.r - rel.Np else 'NOT below'}); objective at the raw truths max {float(rel.res_truth.max()):.1e}")
    return dict(domain=domain, ckpt=ckpt, gate=gate, Xpub=Xpub, ypub=ypub, names=names, true_cls=true_cls, rel=rel, shape=shape, X_raw=X_raw,
                private_idx=perm[: a.N].tolist(), class_pool=lambda c: Xpub[ypub == c])


# ------------------------------------------------------------------------------------------------- one round (global chart)
def global_round(S, chart, a, seed_starts):
    """`--starts` random starts at the public coordinate scale (same seed in every arm); returns results and the candidate ids."""
    rel = S["rel"]; gs = torch.Generator().manual_seed(seed_starts); std = chart.std(S["Xpub"])
    W0s = [(torch.randn(chart.k, 1, generator=gs).to(std.device) * std).reshape(-1) for _ in range(a.starts)]
    t0 = time.time(); res = rel.search(chart, W0s, a.iters)
    cand = pick_candidates(res, rel.N)
    log(f"#     {len(res)} starts in {time.time()-t0:.0f}s ({(time.time()-t0)/len(res):.1f}s/start): objective median {np.median([r['objective'] for r in res]):.2e}, "
        f"min {min(r['objective'] for r in res):.2e}, degenerate {sum(r['degenerate'] for r in res)}, distinct candidates {len(cand)}")
    return res, cand, time.time() - t0


def evaluate(S, chart, res, cand, opt, probs=None):
    """Per-truth records: attacker-matched candidate (the candidate nearest the truth's chart optimum x*_chart) AND the best
       start of any kind (oracle selection). `opt` = oracle_optimum(...) for the N truths in this chart."""
    rel = S["rel"]; X_raw = S["X_raw"]; N = rel.N
    X_found = torch.stack([r["x"] for r in res], 1); X_proj = chart.project(X_raw)
    E = torch.stack([rel_err(X_raw[:, i], X_found) for i in range(N)], 1)           # (starts, N)  recovery-to-TRUTH
    Ep = torch.stack([rel_err(X_proj[:, i], X_found) for i in range(N)], 1)         # (starts, N)  recovery-to-PROJECTION
    Eo = torch.stack([rel_err(opt["X"][:, i], X_found) for i in range(N)], 1)       # (starts, N)  recovery-to-CHART-OPTIMUM x*_chart
    cerr = chart.err(X_raw); obj_proj = rel.obj_of(X_proj)                          # obj_proj = the projection floor in this chart
    per = []
    for i in range(N):
        c = cand[int(Eo[cand, i].argmin())]; b = int(Eo[:, i].argmin())              # matching by distance to x*_chart: the chart's own best
        d = dict(i=i, matched_start=c, err=float(E[c, i]), err_proj=float(Ep[c, i]), err_opt=float(Eo[c, i]), objective=res[c]["objective"],
                 floor=float(obj_proj[i]), objective_opt=opt["objective"][i], err_opt_truth=opt["err_truth"][i], opt_iters=opt["iters"][i],
                 best_any_start=b, best_any_err=float(E[b, i]), best_any_err_proj=float(Ep[b, i]), best_any_err_opt=float(Eo[b, i]), best_any_objective=res[b]["objective"],
                 chart_err_truth=float(cerr[i]), objective_at_truth_raw=float(rel.res_truth[i]), objective_at_truth_projection=float(obj_proj[i]),
                 **flags(float(E[c, i]), float(Ep[c, i]), res[c]["objective"], float(obj_proj[i]), float(Eo[c, i]), opt["objective"][i]))
        if probs is not None:
            p = probs[cand.index(c)]; top = p.topk(2).indices.tolist()
            d.update(top1=S["names"][top[0]], top2=S["names"][top[1]], top1_correct=bool(top[0] == S["true_cls"]), p_true=float(p[S["true_cls"]]))
        per.append(d)
    return per, X_found[:, cand], E


def summary(per):
    nan = float("nan"); g = lambda k: [p[k] if p.get(k) is not None else nan for p in per]
    e, ep, eo, eot, b, c = g("err"), g("err_proj"), g("err_opt"), g("err_opt_truth"), g("best_any_err"), g("chart_err_truth")
    n_opt = sum(1 for p in per if p.get("opt_available"))
    return dict(err_median=float(np.median(e)), err_mean=float(np.mean(e)), err_per_image=e, err_proj_median=float(np.median(ep)), err_proj_per_image=ep,
                err_opt_median=float(np.nanmedian(eo)) if n_opt else nan, err_opt_per_image=eo, err_opt_truth_median=float(np.nanmedian(eot)) if n_opt else nan,
                err_opt_truth_per_image=eot, objective_opt_per_image=g("objective_opt"), n_opt_available=n_opt,
                best_any_err_median=float(np.median(b)), best_any_err_proj_median=float(np.median(g("best_any_err_proj"))),
                chart_err_median=float(np.median(c)), chart_err_max=float(max(c)), chart_err_per_image=c,
                landed=sum(bool(p["landed"]) for p in per), reached_projection=sum(bool(p["reached_projection"]) for p in per),
                alias_flags=sum(bool(p["alias_flag"]) for p in per), solver_short=sum(bool(p["solver_short"]) for p in per), unverified_landings=sum(bool(p["unverified_landing"]) for p in per),
                at_floor=sum(bool(p["at_floor"]) for p in per), reached_opt=sum(bool(p.get("reached_opt")) for p in per), alias_in_chart=sum(bool(p.get("alias_in_chart")) for p in per),
                solver_short_opt=sum(bool(p.get("solver_short_opt")) for p in per), n_obj_above_opt=sum(bool(p.get("obj_above_opt")) for p in per),
                objective_median=float(np.median(g("objective"))), floor_median=float(np.median(g("floor"))), floor_per_image=g("floor"))


def class_decision(S, probs):
    """Round-level class decision: argmax of the summed candidate probabilities; the true class's rank is recorded."""
    tot = probs.sum(0); order = tot.argsort(descending=True).tolist()
    return dict(class_rank=[S["names"][c] for c in order[:5]], rank1=order[0], rank2=order[1], true_class_rank=order.index(S["true_cls"]) + 1,
                top1_acc_on_candidates=float(probs.argmax(1).eq(S["true_cls"]).float().mean()),
                top2_acc_on_candidates=float((probs.topk(2).indices == S["true_cls"]).any(1).float().mean()))


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--releases", nargs="*", default=["mnist", "cifar"])
    ap.add_argument("--mnist-model", default="models/exact_inversion/mnist_mlp_strong.pth"); ap.add_argument("--letter", default="a")
    ap.add_argument("--cifar-ckpt", default="models/exact_inversion/cifar10_cnn_newclass.pth"); ap.add_argument("--cifar-class", default="motorcycle")
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=64); ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--starts", type=int, default=200, help="random starts per global chart (round 0 and every variant-A arm)")
    ap.add_argument("--local-starts", type=int, default=8, help="fresh random starts per candidate per local (variant-B) chart, beside the warm start")
    ap.add_argument("--K", type=int, default=200, help="neighbours per local chart")
    ap.add_argument("--iters", type=int, default=300); ap.add_argument("--a-rounds", type=int, default=2); ap.add_argument("--b-rounds", type=int, default=4)
    ap.add_argument("--clf-epochs", type=int, default=None, help="default 8 (mnist) / 30 (cifar)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data-root", default="data"); ap.add_argument("--mnist-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default="results/bootstrap_chart"); ap.add_argument("--fig-dir", default="figures/bootstrap_chart")
    a = ap.parse_args(); dev = torch.device(a.device)
    os.makedirs(a.save_dir, exist_ok=True); os.makedirs(a.fig_dir, exist_ok=True)
    job = os.environ.get("LSB_JOBID", "local"); a.out = a.out or os.path.join(a.save_dir, f"rounds_{job}.jsonl")
    log(f"# bootstrap_chart  releases={a.releases} N={a.N} r={a.r} k={a.k} T={a.T} K={a.K} starts={a.starts} local_starts={a.local_starts} "
        f"a_rounds={a.a_rounds} b_rounds={a.b_rounds} git={git_hash()} host={socket.gethostname()} job={job} dev={dev}")

    for domain in a.releases:
        S = setup(domain, a, dev); rel = S["rel"]; X_raw = S["X_raw"]; names = S["names"]; true_cls = S["true_cls"]
        common = dict(part="bootstrap_chart", release=domain, backbone=S["ckpt"], base_gate=S["gate"], true_class=names[true_cls], private_idx=S["private_idx"],
                      N=a.N, r=a.r, k=a.k, T=a.T, lr=a.lr, tol=a.tol, K=a.K, iters=a.iters, m=rel.m, n=rel.n, n_prime=rel.Np, rank_C=rel.rank_C,
                      cert_line=a.r - rel.Np, below_cert_line=bool(a.k < a.r - rel.Np), privates="raw", residual_at_truths_max=float(rel.res_truth.max()),
                      seed=a.seed, solver="lm_cert (certificate.py)", release_fn="train_release sgd", git=git_hash(), host=socket.gethostname(), job=job, cmd=" ".join(sys.argv))
        tag = lambda v, t, arm: f"{domain}_{v}_round{t}_{arm}_{job}"
        saved = {}

        def emit(row, tensors, v, t, arm):
            row = dict(**common, variant=v, round=t, arm=arm, **row)
            print(json.dumps(row), flush=True)
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
            p = os.path.join(a.save_dir, tag(v, t, arm) + ".pth"); torch.save(dict(**{kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in tensors.items()}, row=row), p)
            saved[(v, t, arm)] = tensors
            log(f"#   >> {domain} {v} round {t} {arm}: err-to-truth median {row['err_median']:.4f} | err-to-x*chart median {row['err_opt_median']:.4f} (x*chart-to-truth {row['err_opt_truth_median']:.4f}; "
                f"opt available {row['n_opt_available']}/{a.N}) | err-to-projection median {row['err_proj_median']:.4f} | chart err median {row['chart_err_median']:.4f} | "
                f"reached x*chart {row['reached_opt']}/{a.N}, alias-in-chart {row['alias_in_chart']}, solver-short(opt) {row['solver_short_opt']}, obj above opt {row['n_obj_above_opt']} | "
                f"landed {row['landed']}/{a.N}, at proj-floor {row['at_floor']}")

        # ---------------- round 0: the generic public chart (the attacker knows no class)
        log(f"# [{domain}] ROUND 0: generic chart = PCA-{a.k} of ALL {S['Xpub'].shape[0]} public images ({len(names)} classes)")
        chart0 = Chart.pca(S["Xpub"], a.k, "generic_pca"); log(f"#     explained variance {chart0.explained:.3f}; chart error of the truths {[round(float(v), 3) for v in chart0.err(X_raw)]}")
        clf_epochs = a.clf_epochs or (8 if domain == "mnist" else 30)
        clf, cal = train_classifier(domain, S["Xpub"], S["ypub"], len(names), clf_epochs, chart0, dev,
                                    os.path.join(a.save_dir, f"clf_{domain}_k{a.k}_ep{clf_epochs}_s{a.seed}.pth"), a.seed)
        cal_true = cal["val_acc_proj_per_class"][true_cls]
        clf_info = dict(classifier=("784-512-512-26 MLP" if domain == "mnist" else "CNN(m=100)"), clf_epochs=clf_epochs, clf_public_only=True, clf_chance=1.0 / len(names),
                        clf_val_acc_raw=cal["val_acc_raw"], clf_val_acc_projected=cal["val_acc_proj"], clf_val_acc_projected_true_class=cal_true,
                        clf_val_acc_projected_per_class=cal["val_acc_proj_per_class"], clf_calibration_chart=chart0.name, clf_n_val=cal["n_val"])
        log(f"# [{domain}] classifier calibration on PCA-{a.k} projections of held-out public images: overall {cal['val_acc_proj']*100:.1f}%, "
            f"true class '{names[true_cls]}' {cal_true*100:.1f}% (raw images: {cal['val_acc_raw']*100:.1f}%)")
        opt0 = oracle_optimum(rel, chart0, X_raw, a.iters)                            # NOT attacker-available: x*_chart per truth
        log(f"#     chart optimum from the ORACLE start (projection coords): objective {[f'{v:.1e}' for v in opt0['objective']]} vs at projection "
            f"{[f'{v:.1e}' for v in opt0['objective_at_projection']]}; err(x*_chart, truth) {[round(v, 3) for v in opt0['err_truth']]}")
        res0, cand0, sec0 = global_round(S, chart0, a, a.seed + 31)
        probs0 = recognise(clf, torch.stack([res0[c]["x"] for c in cand0], 1))
        per0, Xc0, _ = evaluate(S, chart0, res0, cand0, opt0, probs0); dec0 = class_decision(S, probs0); sm0 = summary(per0)
        n_top1 = sum(p["top1_correct"] for p in per0)                                 # per-truth view: the candidate matched to each truth
        # coordinator 2026-09-18: void only if the solver FELL SHORT of the chart's best -- far from x*_chart AND above its objective
        void_solver = sm0["err_opt_median"] > LAND and sm0["n_obj_above_opt"] > a.N / 2
        void_A = not (n_top1 >= 5 and cal_true > 5 / 8)                                # audit: >= 5/8 top-1 correct AND calibration above that bar
        void = dict(void_round0=bool(void_solver), void_round0_reason=(f"round-0 median err(recovery, x*_chart) {sm0['err_opt_median']:.3e} > 1e-2 AND objective above the oracle-start optimum on {sm0['n_obj_above_opt']}/{a.N} images" if void_solver else ""),
                    void_variant_A=bool(void_A), void_variant_A_reason=(f"round-0 top-1 correct {n_top1}/8 (need >= 5); calibration on projections of the true class {cal_true:.3f} (need > 0.625)" if void_A else ""),
                    round0_top1_correct=n_top1, round0_err_median_above_half=bool(sm0["err_median"] > 0.5), round0_err_proj_median=sm0["err_proj_median"],
                    round0_err_opt_median=sm0["err_opt_median"], round0_alias_in_chart=sm0["alias_in_chart"], round0_solver_short_opt=sm0["solver_short_opt"])
        log(f"# [{domain}] round-0: err(recovery, x*_chart) median {sm0['err_opt_median']:.3e} (reached {sm0['reached_opt']}/{a.N}, alias-in-chart {sm0['alias_in_chart']}, solver-short {sm0['solver_short_opt']}); "
            f"err(x*_chart, truth) median {sm0['err_opt_truth_median']:.3f}; recovery-to-projection median {sm0['err_proj_median']:.3e}; recovery-to-truth median {sm0['err_median']:.3f}; "
            f"recognition: class ranking {dec0['class_rank']}, true class '{names[true_cls]}' rank {dec0['true_class_rank']}, top-1 correct {n_top1}/{a.N} (chance {1/len(names):.3f})")
        log(f"# [{domain}] VOID round0={void['void_round0']} {void['void_round0_reason']} | VOID variant A={void['void_variant_A']} {void['void_variant_A_reason']}")
        common.update(**void, **clf_info)
        row0 = dict(chart=chart0.name, chart_pool="all public classes", chart_pool_size=chart0.pool_size, chart_explained=chart0.explained, starts=len(res0), seconds=sec0,
                    recognition=dec0, per_image=per0, n_candidates=len(cand0), runs=[{kk: vv for kk, vv in r.items() if kk not in ("w", "x")} for r in res0], **sm0)
        emit(row0, dict(x_raw=X_raw, x_chart=chart0.project(X_raw), x_opt=opt0["X"], x_cand=Xc0, x_found=torch.stack([r["x"] for r in res0], 1), chart_mean=chart0.mean, chart_V=chart0.V,
                        cand=cand0, probs=probs0, A_T=rel.A_T, B_T=rel.B_T, C=rel.C), "round0", 0, "generic")
        grid([("truth (raw)", X_raw), ("generic-chart projection of truth", chart0.project(X_raw)), ("x*_chart (oracle start; not attacker-available)", opt0["X"]),
              ("round-0 candidate matched to truth", torch.stack([res0[p["matched_start"]]["x"] for p in per0], 1)),
              ("round-0 best start of any kind", torch.stack([res0[p["best_any_start"]]["x"] for p in per0], 1))], S["shape"], os.path.join(a.fig_dir, tag("round0", 0, "generic") + ".png"),
             f"{domain} round 0, generic PCA-{a.k} chart of all public classes: err-to-truth median {sm0['err_median']:.3f}, err-to-x*chart median {sm0['err_opt_median']:.1e}, "
             f"x*chart-to-truth {sm0['err_opt_truth_median']:.3f}, chart err median {sm0['chart_err_median']:.3f}, recognised {dec0['class_rank'][:2]} (true '{names[true_cls]}' rank {dec0['true_class_rank']}, top-1 {n_top1}/{a.N})")

        # ---------------- variant A: class-recognition bootstrap
        dec = dec0; prev_cls = None
        for t in range(1, a.a_rounds + 1):
            rec_cls, wrong_cls = dec["rank1"], dec["rank2"]
            if rec_cls == prev_cls:
                log(f"# [{domain}] A round {t}: recognised class '{names[rec_cls]}' unchanged -> STABLE, stop"); break
            prev_cls = rec_cls
            arms = [("recognised", rec_cls, True), ("wrong_class", wrong_cls, True), ("oracle_class", true_cls, False)]
            log(f"# [{domain}] A ROUND {t}: recognised '{names[rec_cls]}' (pool {int((S['ypub']==rec_cls).sum())}), wrong-class control '{names[wrong_cls]}', oracle '{names[true_cls]}'")
            charts = {arm: Chart.pca(S["class_pool"](c), a.k, f"class_pca:{names[c]}") for arm, c, _ in arms}
            results = {}
            for arm, c, avail in arms:
                ch = charts[arm]
                twin = next((o for o in results if charts[o].same_as(ch)), None)
                if twin is not None:                                  # the same chart, the same starts: a re-run would reproduce the same numbers exactly
                    res, cand, sec, opt = results[twin]; log(f"#     {arm}: chart identical to '{twin}' -> reusing its (deterministic) results")
                else:
                    log(f"#     {arm}: chart {ch.name} (pool {ch.pool_size}, explained {ch.explained:.3f}), chart err of truths median {float(ch.err(X_raw).median()):.3f}")
                    opt = oracle_optimum(rel, ch, X_raw, a.iters)
                    log(f"#     x*_chart (oracle start): err to truth median {float(np.median(opt['err_truth'])):.3f}, objective median {float(np.median(opt['objective'])):.1e} (at projection {float(np.median(opt['objective_at_projection'])):.1e})")
                    res, cand, sec = global_round(S, ch, a, a.seed + 31)
                results[arm] = (res, cand, sec, opt)
                probs = recognise(clf, torch.stack([res[cc]["x"] for cc in cand], 1))
                per, Xc, _ = evaluate(S, ch, res, cand, opt, probs); d_ = class_decision(S, probs)
                row = dict(chart=ch.name, chart_class=names[c], chart_pool_size=ch.pool_size, chart_explained=ch.explained, attacker_available=avail, identical_to=twin,
                           starts=len(res), seconds=sec, recognition=d_, per_image=per, n_candidates=len(cand), round0_err_median=sm0["err_median"],
                           runs=[{kk: vv for kk, vv in r.items() if kk not in ("w", "x")} for r in res], **summary(per))
                emit(row, dict(x_raw=X_raw, x_chart=ch.project(X_raw), x_opt=opt["X"], x_cand=Xc, x_matched=torch.stack([res[p["matched_start"]]["x"] for p in per], 1),
                               chart_mean=ch.mean, chart_V=ch.V, cand=cand, probs=probs), "A", t, arm)
                if arm == "recognised": dec_next = d_
            g_rows = [("truth (raw)", X_raw)] + [(f"A r{t} {arm} ('{names[c]}')", saved[("A", t, arm)]["x_matched"]) for arm, c, _ in arms]
            grid(g_rows, S["shape"], os.path.join(a.fig_dir, tag("A", t, "all") + ".png"),
                 f"{domain} variant A round {t}: recognised '{names[rec_cls]}' vs wrong-class '{names[wrong_cls]}' vs oracle '{names[true_cls]}' (PCA-{a.k} of each class's public pool)")
            dec = dec_next

        # ---------------- variant B: local-neighbour bootstrap (per candidate slot)
        rec_cls = dec0["rank1"]; pool_B = S["class_pool"](rec_cls); K = min(a.K, pool_B.shape[0])
        log(f"# [{domain}] B: local charts from K={K} neighbours in the recognised class '{names[rec_cls]}' (pool {pool_B.shape[0]}); {a.local_starts} random + 1 warm start per slot")
        slots = {arm: [Xc0[:, j].clone() for j in range(Xc0.shape[1])] for arm in ("recovery", "random_anchor", "oracle_anchor")}
        gr = torch.Generator().manual_seed(a.seed + 77)
        for t in range(1, a.b_rounds + 1):
            for arm in ("recovery", "random_anchor", "oracle_anchor"):
                t0 = time.time(); res_all, per_slot, new, E_b, Ep_b, F_b = [], [], [], [], [], []
                for j, xj in enumerate(slots[arm]):
                    nearest_truth = int(rel_err(xj, X_raw).argmin())
                    anchor = xj if arm == "recovery" else (pool_B[int(torch.randint(pool_B.shape[0], (1,), generator=gr))] if arm == "random_anchor" else X_raw[:, nearest_truth])
                    ch, nb = local_chart(anchor, pool_B, K, a.k, f"local:{arm}")
                    gs = torch.Generator().manual_seed(a.seed + 31 + 1000 * t + j); std = ch.coords(nb.T).std(dim=1, keepdim=True)
                    W0s = [ch.coords(xj[:, None]).reshape(-1)] + [(torch.randn(a.k, 1, generator=gs).to(dev) * std).reshape(-1) for _ in range(a.local_starts)]
                    res = rel.search(ch, W0s, a.iters)
                    best = min([r for r in res if not r["degenerate"]] or res, key=lambda r: r["objective"])
                    new.append(best["x"].clone()); res_all += res
                    X_proj = ch.project(X_raw); floors = rel.obj_of(X_proj); cerr = ch.err(X_raw)
                    E = torch.stack([rel_err(r["x"], X_raw) for r in res]); Ep = torch.stack([rel_err(r["x"], X_proj) for r in res])   # (starts, N)
                    eb, epb = rel_err(best["x"], X_raw), rel_err(best["x"], X_proj); i = int(epb.argmin())
                    opt = oracle_optimum(rel, ch, X_raw[:, i:i + 1], a.iters)          # x*_chart of the slot's nearest truth in THIS local chart (1 oracle start)
                    e_opt = float(rel_err(best["x"], opt["X"])[0])
                    E_b.append(eb); Ep_b.append(epb); F_b.append(floors)
                    per_slot.append(dict(slot=j, nearest_truth=i, err=float(eb[i]), err_proj=float(epb[i]), err_opt=e_opt, objective=best["objective"], floor=float(floors[i]),
                                         objective_opt=opt["objective"][0], err_opt_truth=opt["err_truth"][0], opt_iters=opt["iters"][0], objective_at_projection=opt["objective_at_projection"][0],
                                         warm_objective=res[0]["objective"], warm_err_proj=float(rel_err(res[0]["x"], X_proj)[i]), best_is_warm=bool(best is res[0]),
                                         chart_err_truth=float(cerr[i]), chart_err_all=[float(v) for v in cerr], chart_err_prev_candidate=float(ch.err(xj[:, None])[0]),
                                         best_any_err=float(E[:, i].min()), best_any_err_proj=float(Ep[:, i].min()), best_any_err_opt=float(min(rel_err(r["x"], opt["X"])[0] for r in res)), n_starts=len(res),
                                         anchor_err_vs_truth=float(rel_err(anchor, X_raw)[i]), **flags(float(eb[i]), float(epb[i]), best["objective"], float(floors[i]), e_opt, opt["objective"][0])))
                slots[arm] = new
                Xn = torch.stack(new, 1); probs = recognise(clf, Xn)
                E_b, Ep_b, F_b = torch.stack(E_b), torch.stack(Ep_b), torch.stack(F_b)                        # (slots, N)
                per = []
                for i in range(a.N):
                    # per-truth view: among slots whose nearest truth is i, the one nearest its x*_chart; if no slot points at truth i,
                    # the slot nearest its projection, with the optimum-based columns marked unavailable (no oracle start was run for it)
                    mine = [p for p in per_slot if p["nearest_truth"] == i]
                    if mine: ps = min(mine, key=lambda p: p["err_opt"]); j = ps["slot"]; eo, oo, eot = ps["err_opt"], ps["objective_opt"], ps["err_opt_truth"]
                    else: j = int(Ep_b[:, i].argmin()); ps = per_slot[j]; eo = oo = eot = None
                    top = probs[j].topk(2).indices.tolist()
                    per.append(dict(i=i, matched_slot=j, slot_points_at_truth=bool(mine), err=float(E_b[j, i]), err_proj=float(Ep_b[j, i]), err_opt=eo, objective=ps["objective"], floor=float(F_b[j, i]),
                                    objective_opt=oo, err_opt_truth=eot, chart_err_truth=ps["chart_err_all"][i], best_any_err=min(float(rel_err(r["x"], X_raw)[i]) for r in res_all),
                                    best_any_err_proj=float(Ep_b[:, i].min()), objective_at_truth_raw=float(rel.res_truth[i]),
                                    top1=names[top[0]], top2=names[top[1]], top1_correct=bool(top[0] == true_cls), **flags(float(E_b[j, i]), float(Ep_b[j, i]), ps["objective"], float(F_b[j, i]), eo, oo)))
                sm = summary(per); sm["chart_err_per_slot"] = [p["chart_err_truth"] for p in per_slot]
                row = dict(chart=f"local_pca:{arm}", chart_class=names[rec_cls], chart_pool_size=pool_B.shape[0], K_used=K, attacker_available=(arm != "oracle_anchor"),
                           local_starts=a.local_starts, starts=len(res_all), seconds=time.time() - t0, per_slot=per_slot, per_image=per, recognition=class_decision(S, probs),
                           round0_err_median=sm0["err_median"],
                           slots_distinct=len(pick_candidates([dict(x=x, objective=p["objective"], degenerate=False) for x, p in zip(new, per_slot)], a.N)), **sm)
                emit(row, dict(x_raw=X_raw, x_slots=Xn, probs=probs), "B", t, arm)
            grid([("truth (raw)", X_raw), (f"B r{t} recovery-anchored", saved[("B", t, "recovery")]["x_slots"]), (f"B r{t} random-anchor control", saved[("B", t, "random_anchor")]["x_slots"]),
                  (f"B r{t} oracle truth-anchor", saved[("B", t, "oracle_anchor")]["x_slots"])], S["shape"], os.path.join(a.fig_dir, tag("B", t, "all") + ".png"),
                 f"{domain} variant B round {t}: local PCA-{a.k} of K={K} neighbours in '{names[rec_cls]}' -- slot columns are candidate slots, not matched to truth columns")
        log(f"# [{domain}] done.")


if __name__ == "__main__":
    main()
