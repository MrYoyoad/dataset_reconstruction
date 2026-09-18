#!/usr/bin/env python3
"""Multilayer rank law on a CONVOLUTIONAL frozen encoder, plus the T arm (plan 2026-09-18, WP1 as amended by the
Audit 2026-09-18 section: bottleneck spec, r = 256, six adaptable modules, first in {1, 3, 5}).

Port of `real_encoder_ranklaw.py` (same row format, same tolerance ladder, same gap diagnostic, same
pre-registration) onto a conv backbone from `conv_certificate.py`: `SPECS["bottleneck"]` = conv 1->64->128->8->256
(k=3, stride 2, pad 1, GELU; spatial 28->14->7->4->2) -> dense 1024->1000 (GELU) -> head 10, checkpoint
`models/exact_inversion/mnist_conv_bottleneck.pth`.  Any spec in `conv_certificate.SPECS` runs (the deep spec is the
SMOKE-ONLY code-path exercise; it is vacuous by arithmetic, see the audit).

WHAT CHANGES ON A CONV LAYER.  The object a shared kernel multiplies is the PATCH MATRIX P_l(x) = unfold(h_l) of
shape (p_l, P_l), p_l = C_in*3*3 the patch dimension, P_l the number of positions.  A recorded image contributes
P_l patch vectors, so the recorded span U_l is the column space of the N*P_l patch vectors of the truths and the
recorded count N'_l is its rank (`patch_span_rank`) -- NOT N.  The zero-drift certificate is the MLP construction,
C_l = P_{col(A_{l,0} U_l)^perp} A_{l,0}, of rank min(r, p_l) - N'_l, and the stacked objective applies it at EVERY
position: g(v) = concat_l vec(C_l P_l(psi(v))).  The dense hidden layer and the head have P = 1 and are treated
exactly as in the MLP harness.  Symbols on the row (audit): `p_l` = patch/input dimension, `d_j` = rank M_l (the
Jacobian rank of vec(P_l) w.r.t. the chart coordinate, T5's symbol), `q_l` = rank((C_l (x) I_{P_l}) J_l).
T5.2's per-layer budget generalised to weight sharing is q_l_formula = min(rank(C_l)*P_l, d_l); for P_l = 1 it is
min(r - N', d_l).  Conv vacuity flag: patch_span_rank >= min(r, p_l).  The invariant "N' = N at the first adapted
layer at every T" is asserted for DENSE modules only (a conv first layer records its patch span, never N).
sigma0 is set PER LAYER as 1/sqrt(p_l).

HOW THE STACKED JACOBIAN IS BUILT.  J_l = d vec(P_l(psi(v)))/dv is computed once per (chart, image) with
`torch.func.jacfwd` (chunked through vmap(jvp) if `--jac-chunk` is set; sizes printed in the header).  The
certificate is LINEAR in the patches, so (C_l (x) I) J_l IS the Jacobian of vec(C_l P_l) (chain rule for a linear
map) -- one differentiation serves d_l, q_l, the zero-drift stack and every T-arm stack.  Certificates that are
numerically zero (absolute floor 1e-10 * sigma_max(A): a relative threshold calls a zero matrix full rank, LESSONS)
are VACUOUS and contribute no rows (`layers_dropped_vacuous`).  Everything is per image, then the median.

THE T ARM (`--T-arm 1 5 20 100 400`, lr 0.01, full batch, B_0 = 0, FP64).  LoRA is trained on ALL adapted modules
of the config.  `conv_certificate.run_training` has no LoRA slot on the dense hidden layer (it is frozen encoder
there), so this file carries `run_training6`, a line-for-line mirror with a dense slot; at load it is CHECKED to
reproduce `conv_certificate.run_training` to machine precision when the dense slot is None (printed; the job
aborts otherwise).  The drifted certificate C_l = P_{row(B_{l,T})^perp} A_{l,T} is read from the release and per
module the row records rank B_{l,T} (= N'_l(T)), rank C_l, the residual ||C_l h|| / ||A_{l,T} h|| on the FROZEN
base patches (what an attacker can compute), and the stacked ladder with the drifted certificates.  Hypotheses
(audit): conv modules -- does N'_l(T) move with T at all (B_{l,1} already sums N*P_l rank-one terms at T = 1);
dense modules -- N' = N*T growth (ledger M2, synthetic) versus a plateau (the r=256 MLP plateau); the row decides.

PRE-REGISTERED, before any row (WP1 (a)-(c) + audit):
  DISCRIMINATION:  in a config with corrected_pred < t52_pred, measured rank at corrected_pred, strictly below k_1.
  FALSIFIED:       measured at t52_pred in such a config.
  CONTROL:         first = 5 (dense + head only) has corrected_pred == t52_pred and must saturate there.
  VACUOUS:         no config has corrected_pred < t52_pred (a legitimate result).
  CONV-VACUOUS per module: rank C_l = 0 where patch_span_rank >= min(r, p_l) -- recorded, never hidden.
  TWO ARITHMETIC PREDICTIONS are printed in the header from the MEASURED patch spans, because they disagree on
  whether this spec discriminates: (i) the weight-sharing count q_l = min(rank(C_l)*P_l, d_l) -- then conv 2
  alone (24 x 49 = 1176 >= k) pins the whole chart and the laws COINCIDE at every config; (ii) the audit's
  dense-style count q_l = rank(C_l) -- then corrected (= d_conv4 + q_1 + q_2 + q_3 ~ 280) < T5.2 (~449) for
  k > 280 at first = 1.  The measured q_l (against q_l_formula) settles which count is real BEFORE the law is read.
  No gap at the cut -> ladder and spectrum are reported, never an integer.
  WP0: every row carries the frozen base's train acc / loss / test acc, read from the checkpoint dict when present
       AND measured at load on the full splits; `fully_trained_gate` = train acc >= 99.5% and train loss <= 1e-2.

PROGRAM FLAGS (plan 2026-09-18, P1/P4; added 2026-09-18; default behaviour byte-identical, new fields only).
  `--layers` takes the same layer-selection PATTERNS as `real_encoder_ranklaw.py` (prefix | alternate | suffix:<L> |
  middle:<L> | random:<L>:<n> | single:<l> | explicit:<l1,..>; single/explicit take ABSOLUTE 1-indexed module
  numbers and are skipped with a note when a module lies before `first`), every RANKLAW row carries `layer_pattern`
  / `pattern_draw`, and both law predictions are computed over the CHOSEN modules in network order (k_1 = the
  shallowest chosen module's d).  For non-prefix patterns the corrected law's NESTING READING is what is being
  tested (plan P4), exactly as documented there.  The modules that receive A0 / a certificate / LoRA training in
  the T arm are the UNION of the modules any pattern uses (`layers_adapted` on the row; = the maxL prefix under the
  default).  A0 is drawn for every adaptable module in network order from one generator per (seed, first), so
  A0[l] is the same under every pattern.  Condition-number fields `cond_at_1e10` / `cond_at_corrected` /
  `cond_at_fp16` (+ `sigma_at_rank_1e10`, `rank_1e10_per_image`), `--seed` as a LIST (truths, A_0, random draws
  and the run_training6 check are per seed; `seed` on the config key and on PATCH_SPAN rows), r as the INNERMOST
  loop (the patch Jacobians and d_j are r-independent and were already shared), and the THIRD outcome
  `rank_test_outcome` / `rank_verdict` (compare | no_gap_vacuous | dead; a stack with no rows is no_gap_vacuous)
  as in the MLP harness.

  python -u -m experiments.multilayer_cert.conv_encoder_ranklaw --spec bottleneck \
      --ckpt models/exact_inversion/mnist_conv_bottleneck.pth --r 256 --first 1 3 5 --maxL 6 --T-arm 1 5 20 100 400
"""
import argparse, json, math, os, socket, sys, time
import torch, torch.func as tf, torch.nn.functional as F

import experiments.exact_inversion.conv_certificate as cc
from experiments.exact_inversion.certificate import certificate
from experiments.exact_inversion.trained_backbone import PCAChart, read_idx
from experiments.multilayer_cert.common import provenance
from experiments.multilayer_cert.real_encoder_ranklaw import (LADDER, TOL, med, numrank, zero_drift_cert,
                                                              layer_selections, cond_fields, rank_outcome)

torch.set_default_dtype(torch.float64)
RANK_TOL = 1e-10           # the clean-rank cut for d_l, q_l, cert ranks (as in the MLP harness)
GATE_TRAIN_ACC, GATE_TRAIN_LOSS = 0.995, 1e-2
ARITH_KS = (128, 256, 384, 784)


# ----------------------------------------------------------------------------------------------------------------
# geometry: one entry per ADAPTABLE module (convs, optional dense hidden, head)
def spec_geometry(spec, dense_hidden, side0=28, n_classes=10):
    geo, side, rep = [], side0, side0 * side0
    for (ci, co, k, s, p) in spec:
        hout = (side + 2 * p - k) // s + 1
        geo.append(dict(kind="conv", cin=ci, cout=co, k=k, s=s, p=p, hin=side, hout=hout,
                        p_l=ci * k * k, P=hout * hout, in_rep=rep, out_dim=co * hout * hout))
        side, rep = hout, co * hout * hout
    if dense_hidden:
        geo.append(dict(kind="dense", p_l=rep, P=1, in_rep=rep, out_dim=dense_hidden, cout=dense_hidden))
        rep = dense_hidden
    geo.append(dict(kind="head", p_l=rep, P=1, in_rep=rep, out_dim=n_classes, cout=n_classes))
    for l, gg in enumerate(geo):                                # narrowest representation upstream of this input
        gg["width_bound"] = min(g2["in_rep"] for g2 in geo[:l + 1])
    return geo


def unfold_index(H, W, k, s, p, dev):
    """Gather index reproducing F.unfold's row order (c*k*k + ki*k + kj) and column order (oy*Wo + ox)."""
    Hp, Wp = H + 2 * p, W + 2 * p
    Ho, Wo = (Hp - k) // s + 1, (Wp - k) // s + 1
    oy, ox = torch.meshgrid(torch.arange(Ho), torch.arange(Wo), indexing="ij")
    ki, kj = torch.meshgrid(torch.arange(k), torch.arange(k), indexing="ij")
    rows = oy.reshape(1, -1) * s + ki.reshape(-1, 1)                  # (k*k, P)
    cols = ox.reshape(1, -1) * s + kj.reshape(-1, 1)
    return (rows * Wp + cols).to(dev)


def patches_single(h, idx, k, p):
    """(C, H, W) -> (C*k*k, P): unfold as pad + gather, forward-AD and vmap friendly."""
    hp = F.pad(h, (p, p, p, p))
    C = hp.shape[0]
    return hp.reshape(C, -1)[:, idx].reshape(C * k * k, -1)


def patch_maps(x_vec, Wms, bs, Wd, bd, geo, idxs):
    """Pixel vector (784,) -> the FROZEN input of every adaptable module: [P_1..P_Lc, flat(, dense_out)]."""
    h = x_vec.reshape(1, geo[0]["hin"], geo[0]["hin"])
    outs = []
    for l, g in enumerate([gg for gg in geo if gg["kind"] == "conv"]):
        P = patches_single(h, idxs[l], g["k"], g["p"])
        outs.append(P)
        z = Wms[l] @ P + bs[l][:, None]
        h = F.gelu(z.reshape(g["cout"], g["hout"], g["hout"]))
    flat = h.reshape(-1)
    outs.append(flat[:, None])
    if Wd is not None:
        outs.append(F.gelu(Wd @ flat + bd)[:, None])
    return outs


def forward6(x, Wms, bs, Whead, bhead, Wd, bd, As, Bs):
    """conv_certificate.conv_forward with one extra LoRA slot on the dense hidden layer.  Module order of As/Bs:
       convs, (dense if Wd is not None), head.  Mirrors the reference op for op."""
    h = x; nconv = len(cc.SPEC)
    for l, (cin, cout, k, s, p) in enumerate(cc.SPEC):
        P = cc.patches_of(h, k, s, p)
        z = Wms[l] @ P + bs[l][:, None]
        if As[l] is not None: z = z + Bs[l] @ (As[l] @ P)
        side = int(math.isqrt(z.shape[-1]))
        h = F.gelu(z.reshape(z.shape[0], cout, side, side))
    flat = h.reshape(h.shape[0], -1)
    if Wd is not None:
        zd = flat @ Wd.T + bd
        if As[nconv] is not None: zd = zd + (flat @ As[nconv].T) @ Bs[nconv].T
        flat = F.gelu(zd)
    z = flat @ Whead.T + bhead
    if As[-1] is not None: z = z + (flat @ As[-1].T) @ Bs[-1].T
    return z


def run_training6(x, Wms, bs, Whead, bhead, Wd, bd, A0s, y, m, T, lr):
    """conv_certificate.run_training, line for line, with the dense slot (verified against it at load)."""
    outs = [c for _, c, _, _, _ in cc.SPEC] + ([Wd.shape[0]] if Wd is not None else []) + [m]
    As = [None if a is None else a.detach().clone().requires_grad_(True) for a in A0s]
    Bs = [None if a is None else torch.zeros(o, a.shape[0], dtype=x.dtype, device=x.device, requires_grad=True)
          for o, a in zip(outs, A0s)]
    live = [i for i, a in enumerate(As) if a is not None]
    Y = torch.eye(m, device=x.device)[y]
    for _ in range(T):
        z = forward6(x, Wms, bs, Whead, bhead, Wd, bd, As, Bs)
        zs = z - z.max(dim=1, keepdim=True).values
        p = torch.exp(zs); p = p / p.sum(dim=1, keepdim=True)
        loss = -(Y * torch.log(p + 1e-300)).sum() / x.shape[0]
        params = [As[i] for i in live] + [Bs[i] for i in live]
        gs = torch.autograd.grad(loss, params)
        n = len(live)
        for j, i in enumerate(live):
            As[i] = (As[i] - lr * gs[j]).detach().requires_grad_(True)
            Bs[i] = (Bs[i] - lr * gs[n + j]).detach().requires_grad_(True)
    return ([None if a is None else a.detach() for a in As],
            [None if b is None else b.detach() for b in Bs])


def jac_patches(flat_fn, v, chunk):
    """d flat_fn / dv, (rows, k).  chunk <= 0 or >= k: one torch.func.jacfwd call; else vmap(jvp) over chunks of
       the tangent basis (identical numerics, bounded memory)."""
    kdim = v.numel()
    if chunk <= 0 or chunk >= kdim:
        return tf.jacfwd(flat_fn)(v).detach()
    E = torch.eye(kdim, device=v.device)
    cols = []
    for s0 in range(0, kdim, chunk):
        t = E[s0:s0 + chunk]
        cols.append(tf.vmap(lambda tt: tf.jvp(flat_fn, (v,), (tt,))[1])(t))   # (chunk, rows)
    return torch.cat(cols, 0).T.detach()


def apply_cert(C, J_l, p_l, P_l):
    """(C (x) I_P) J_l: rows of J_l are vec(P_l) in (p_l, P_l) row-major order -> (r*P_l, k) in (r, P_l) order."""
    kdim = J_l.shape[1]
    return (C @ J_l.reshape(p_l, P_l * kdim)).reshape(C.shape[0] * P_l, kdim)


def abs_rank(M, scale, tol=RANK_TOL):
    """Rank with an ABSOLUTE floor tol*scale (scale = sigma_max of the matrix that set M's size)."""
    if M.numel() == 0: return 0
    sv = torch.linalg.svdvals(M)
    return int((sv > tol * scale).sum())


class SubChart:
    """The first k columns of one PCA basis: identical math to PCAChart(Xtrain, k) (nested basis), one SVD."""
    def __init__(self, big, k):
        self.mean, self.V, self.k = big.mean, big.V[:, :k].contiguous(), k
    def psi(self, W): return self.mean[:, None] + self.V @ W
    def coords_of(self, X): return self.V.T @ (X - self.mean[:, None])


@torch.no_grad()
def base_stats(X, y, fwd, chunk=2000):
    """Full-split accuracy and mean cross-entropy of the frozen backbone (FP64), the WP0 gate numbers."""
    correct, loss = 0, 0.0
    for i in range(0, X.shape[0], chunk):
        z = fwd(X[i:i + chunk]); yb = y[i:i + chunk]
        loss += float(F.cross_entropy(z, yb, reduction="sum")); correct += int((z.argmax(1) == yb).sum())
    return correct / X.shape[0], loss / X.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/exact_inversion/mnist_conv_bottleneck.pth")
    ap.add_argument("--spec", default="bottleneck", choices=sorted(cc.SPECS))
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", nargs="*", type=int, default=[256], help="LoRA ranks (audit: 256 everywhere)")
    ap.add_argument("--ks", nargs="*", type=int, default=[16, 32, 66, 128, 256, 384, 512, 784],
                    help="chart dimensions; k >= 784 is the pixel chart")
    ap.add_argument("--first", nargs="*", type=int, default=[1, 3, 5], help="1-indexed first adapted module per config")
    ap.add_argument("--maxL", type=int, default=6, help="max adapted modules stacked from the first (4 convs + dense + head)")
    ap.add_argument("--layers", nargs="*", default=["prefix"],
                    help="layer-selection patterns (P4), as in real_encoder_ranklaw.py: prefix | alternate | suffix:<L> | "
                         "middle:<L> | random:<L>:<n> | single:<l> | explicit:<l1,l2,...>")
    ap.add_argument("--T-arm", nargs="*", type=int, default=[], help="LoRA step counts for the drifted-certificate arm")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None, help="A_0 scale for every layer; default = PER LAYER 1/sqrt(p_l) (audit)")
    ap.add_argument("--seed", nargs="*", type=int, default=[1], help="seeds (truths, A_0, random layer draws); one config per seed")
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--jac-chunk", type=int, default=0, help="tangents per forward-mode chunk; 0 = single jacfwd")
    ap.add_argument("--label", default="", help="free-text run label carried on every row (e.g. smoke-only)")
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    PROV = provenance(__file__)
    cc.SPEC = cc.SPECS[a.spec]                                  # conv_certificate's forward/training read this global
    m = 10

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    # ---- data, backbone, WP0 base-training numbers ----------------------------------------------------------
    Xtr, ytr = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr, device=dev); Xte_t = torch.tensor(Xte, device=dev)
    ytr_t = torch.tensor(ytr, device=dev); yte_t = torch.tensor(yte, device=dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    if ck.get("spec") is not None and ck["spec"] != a.spec:
        raise SystemExit(f"checkpoint {a.ckpt} was trained as spec {ck['spec']!r}, not {a.spec!r}")
    Wms = [w.to(dev).double() for w in ck["Wms"]]; bs_ = [b.to(dev).double() for b in ck["bs"]]
    Whead = ck["Whead"].to(dev).double(); bhead = ck["bhead"].to(dev).double()
    Wd = ck["Wd"].to(dev).double() if ck.get("Wd") is not None else None
    bd = ck["bd"].to(dev).double() if Wd is not None else None
    dense_hidden = Wd.shape[0] if Wd is not None else None
    geo = spec_geometry(cc.SPEC, dense_hidden)
    D = len(geo); nconv = len(cc.SPEC)
    assert len(Wms) == nconv and all(w.shape == (g["cout"], g["p_l"]) for w, g in zip(Wms, geo[:nconv])), \
        f"checkpoint {a.ckpt} does not match spec {a.spec}"
    assert Whead.shape[1] == geo[-1]["p_l"], f"head input {Whead.shape[1]} != geometry {geo[-1]['p_l']}"
    fwd = (lambda X, As=None, Bs=None: cc.conv_forward(X, Wms, bs_, Whead, bhead, As, Bs, Wd=Wd, bd=bd))
    tr_acc, tr_loss = base_stats(Xtr_t.reshape(-1, 1, 28, 28), ytr_t, fwd)
    te_acc, te_loss = base_stats(Xte_t.reshape(-1, 1, 28, 28), yte_t, fwd)
    BASE = dict(ckpt=a.ckpt, spec=a.spec, dense_hidden=dense_hidden, ckpt_keys=sorted(ck.keys()),
                ckpt_test_acc=ck.get("test_acc"), ckpt_train_acc=ck.get("train_acc"),
                ckpt_train_loss=ck.get("train_loss"), ckpt_epochs=ck.get("epochs_run", ck.get("epochs")),
                ckpt_git=ck.get("git"),
                measured_train_acc=tr_acc, measured_train_loss=tr_loss,
                measured_test_acc=te_acc, measured_test_loss=te_loss,
                fully_trained_gate=bool(tr_acc >= GATE_TRAIN_ACC and tr_loss <= GATE_TRAIN_LOSS),
                gate=f"train acc >= {GATE_TRAIN_ACC} and train loss <= {GATE_TRAIN_LOSS} (WP0)", label=a.label)
    print(f"# conv_encoder_ranklaw spec={a.spec} ckpt={a.ckpt} modules={[g['kind'] for g in geo]} N={a.N} r={a.r} "
          f"ks={a.ks} first={a.first} maxL={a.maxL} layers={a.layers} T_arm={a.T_arm} lr={a.lr} label={a.label!r} "
          f"git={PROV['git']} script_sha={PROV['script_sha']} dev={dev}", flush=True)
    print(f"# BASE (WP0): ckpt test_acc={BASE['ckpt_test_acc']} ckpt train_acc={BASE['ckpt_train_acc']} "
          f"ckpt train_loss={BASE['ckpt_train_loss']} | MEASURED train acc {tr_acc*100:.3f}% loss {tr_loss:.3e} "
          f"test acc {te_acc*100:.2f}% loss {te_loss:.3e} | fully_trained_gate={BASE['fully_trained_gate']}",
          flush=True)
    print("# THEORY TEST of the rank law on a conv encoder, NOT an attack config", flush=True)

    SEEDS = {}
    npix = Xte_t.shape[1]
    for seed in a.seed:                                          # per-seed truths, spans, certificates, T arms
        # ---- truths (same selection recipe as conv_certificate / real_encoder_ranklaw: seed+7) ------------------
        g = torch.Generator().manual_seed(seed + 7)
        idx = torch.randperm(Xte_t.shape[0], generator=g)[:a.N].to(dev)
        X_real = Xte_t[idx].T.contiguous()                          # (784, N)
        X_img = X_real.T.reshape(a.N, 1, 28, 28).contiguous(); y = yte_t[idx]
        print(f"# private join-key indices: {idx.tolist()} labels {y.tolist()} (seed {seed})", flush=True)

        # ---- base inputs of every module at the truths, recorded spans, gather-unfold check ----------------------
        idxs = [unfold_index(gg["hin"], gg["hin"], gg["k"], gg["s"], gg["p"], dev) for gg in geo[:nconv]]
        with torch.no_grad():
            _, kept = cc.conv_forward(X_img, Wms, bs_, Whead, bhead, want_patches=True, Wd=Wd, bd=bd)
            pm = [patch_maps(X_real[:, i], Wms, bs_, Wd, bd, geo, idxs) for i in range(a.N)]
        Hbase = [kept[l].permute(1, 0, 2).reshape(geo[l]["p_l"], -1).contiguous() for l in range(nconv)]
        Hbase += [torch.cat([pm[i][l] for i in range(a.N)], 1).contiguous() for l in range(nconv, D)]
        impl_err = max(float((pm[i][l] - kept[l][i]).abs().max()) for l in range(nconv) for i in range(a.N))
        with torch.no_grad():                                        # dense/head inputs cross-checked against the reference forward
            h = X_img
            for l, gg in enumerate(geo[:nconv]):
                h = F.gelu((Wms[l] @ kept[l] + bs_[l][:, None]).reshape(a.N, gg["cout"], gg["hout"], gg["hout"]))
            flat = h.reshape(a.N, -1)
            impl_err = max(impl_err, float((flat.T - Hbase[nconv]).abs().max()))
            if Wd is not None:
                impl_err = max(impl_err, float((F.gelu(flat @ Wd.T + bd).T - Hbase[nconv + 1]).abs().max()))
        assert impl_err < 1e-13, f"gather-unfold path disagrees with conv_certificate.conv_forward: {impl_err}"
        print(f"# gather-unfold vs conv_certificate.conv_forward inputs: max abs diff {impl_err:.1e}", flush=True)
        H0_norm = [float(torch.linalg.norm(Hbase[l], dim=0).median()) for l in range(D)]
        underflow = [bool(H0_norm[l] < 1e-290) for l in range(D)]
        Ubase, nprime_zd = [], []
        for l in range(D):
            U, S, _ = torch.linalg.svd(Hbase[l], full_matrices=False)
            rk = int((S > RANK_TOL * S[0]).sum()); Ubase.append(U[:, :rk].contiguous()); nprime_zd.append(rk)
            emit(dict(part="PATCH_SPAN", seed=seed, layer=l + 1, kind=geo[l]["kind"], p_l=geo[l]["p_l"], P_l=geo[l]["P"],
                      n_patch_vectors=a.N * geo[l]["P"], patch_span_rank=rk, width_bound=geo[l]["width_bound"],
                      spans_input_dim=bool(rk >= geo[l]["p_l"]),
                      conv_vacuous_at_r={str(r): bool(rk >= min(r, geo[l]["p_l"])) for r in a.r},
                      cert_rank_formula_at_r={str(r): max(0, min(r, geo[l]["p_l"]) - rk) for r in a.r},
                      dense_invariant_ok=(bool(rk == a.N) if geo[l]["kind"] != "conv" else None),
                      H0_norm=H0_norm[l], H0_underflow=underflow[l], N=a.N, base=BASE, git=PROV["git"],
                      script_sha=PROV["script_sha"]))
            print(f"  module {l+1} ({geo[l]['kind']}): p_l={geo[l]['p_l']} P_l={geo[l]['P']} N*P={a.N*geo[l]['P']} "
                  f"out_dim={geo[l]['out_dim']} width_bound={geo[l]['width_bound']} patch span rank N'={rk}"
                  + (f"  dense invariant N'==N: {rk == a.N}" if geo[l]["kind"] != "conv" else ""), flush=True)

        # ---- the discriminating-regime arithmetic from the MEASURED spans, both counts ----------------------------
        print("# DISCRIMINATING-REGIME ARITHMETIC from the measured patch spans.  d_j(pred) = min(k, width_bound_j) "
              f"(width bounds {[g['width_bound'] for g in geo]}).  Two counts: (i) weight-sharing q_l = min(rank C_l * P_l, d_l); "
              "(ii) audit dense-style q_l = min(rank C_l, d_l).", flush=True)
        for r in a.r:
            cr = [max(0, min(r, geo[l]["p_l"]) - nprime_zd[l]) for l in range(D)]
            print(f"#   r={r}: cert rank min(r,p_l)-N'_l = {cr}; vacuous modules {[l+1 for l in range(D) if cr[l] == 0]}", flush=True)
            for first in a.first:
                adapted = list(range(first - 1, min(D, first - 1 + a.maxL)))
                for k in ARITH_KS:
                    k_eff = min(k, npix)
                    dpred = [min(k_eff, geo[l]["width_bound"]) for l in adapted]
                    for tag, qpred in (("(i) xP", [min(cr[l] * geo[l]["P"], dp) for l, dp in zip(adapted, dpred)]),
                                       ("(ii) dense-style", [min(cr[l], dp) for l, dp in zip(adapted, dpred)])):
                        Sq = sum(qpred); t52 = min(dpred[0], Sq)
                        corr = min([dpred[j] + sum(qpred[:j]) for j in range(len(adapted))] + [Sq])
                        print(f"#   r={r} first={first} L={len(adapted)} k={k_eff} {tag}: d={dpred} q={qpred} sum={Sq} "
                              f"T5.2={t52} corrected={corr} discriminates={corr < t52}", flush=True)
        rows_patch = sum(gg["p_l"] * gg["P"] for gg in geo)
        per_tan = npix + sum(gg["p_l"] * gg["P"] + 2 * gg["out_dim"] for gg in geo[:nconv]) + sum(gg["p_l"] for gg in geo[nconv:])
        kmax_rows = max(sum(r * gg["P"] for gg in geo) for r in a.r)
        print(f"# JACOBIAN SIZES: patch Jacobian {rows_patch} x k ({rows_patch*npix*8/1e6:.0f} MB at k=784, one image at a "
              f"time); stacked <= {kmax_rows} x k ({kmax_rows*npix*8/1e6:.0f} MB); forward-mode intermediates "
              f"~{per_tan*8/1e6:.2f} MB per tangent -> {per_tan*8*npix/1e9:.2f} GB for k=784 in one chunk "
              f"(jac_chunk={a.jac_chunk})", flush=True)

        # ---- verify run_training6 == conv_certificate.run_training when the dense slot is None -------------------
        if a.T_arm:
            gchk = torch.Generator().manual_seed(seed + 99)
            rchk = min(a.r)
            A0chk = [(torch.randn(rchk, geo[l]["p_l"], generator=gchk) / math.sqrt(geo[l]["p_l"])).to(dev)
                     for l in range(nconv)] + ([None] if Wd is not None else []) + [
                     (torch.randn(rchk, geo[-1]["p_l"], generator=gchk) / math.sqrt(geo[-1]["p_l"])).to(dev)]
            A_ref, B_ref = cc.run_training(X_img, Wms, bs_, Whead, bhead, [x for x in A0chk if x is not None], y, m, 3, a.lr,
                                           Wd=Wd, bd=bd)
            A_new, B_new = run_training6(X_img, Wms, bs_, Whead, bhead, Wd, bd, A0chk, y, m, 3, a.lr)
            A_new = [x for x in A_new if x is not None]; B_new = [x for x in B_new if x is not None]
            chk = max(max(float((p - q).abs().max() / (q.abs().max() + 1e-300)) for p, q in zip(A_new, A_ref)),
                      max(float((p - q).abs().max() / (q.abs().max() + 1e-300)) for p, q in zip(B_new, B_ref)))
            print(f"# run_training6 vs conv_certificate.run_training (dense slot None, T=3): max rel diff {chk:.1e}", flush=True)
            assert chk < 1e-12, f"run_training6 does not reproduce conv_certificate.run_training: {chk}"

        # ---- certificates per (r, first): zero drift + T arm (chart independent) ----------------------------------
        sig = [a.sigma0 if a.sigma0 is not None else 1.0 / math.sqrt(geo[l]["p_l"]) for l in range(D)]
        print(f"# sigma0 per module = {'given' if a.sigma0 is not None else '1/sqrt(p_l)'}: {[round(s, 5) for s in sig]}", flush=True)
        CONF = {}
        with torch.no_grad():
            z0 = fwd(X_img); loss0 = float(F.cross_entropy(z0, y)); acc0 = float((z0.argmax(1) == y).double().mean())
            # imprint strength: B_1 = -lr * sum_i delta_i (A0 h_i)^T with delta_i proportional to the softmax residual
            # p_i - y_i, so an image the frozen base already classifies confidently imprints below the certificate's
            # 1e-12 cut and rank B_T counts N' < N on a DENSE module without any drift -- recorded, so a dense N' < N
            # can be read against it instead of being taken for a harness bug
            resid = torch.linalg.norm(torch.softmax(z0, 1) - torch.eye(m, device=dev)[y], dim=1)
            imprint_rel = (resid / resid.max()).tolist()
            n_imprint_1e12 = int(sum(1 for v in imprint_rel if v > 1e-12)); n_imprint_1e10 = int(sum(1 for v in imprint_rel if v > 1e-10))
        print(f"# base softmax residual per image (relative to max) {[f'{v:.1e}' for v in imprint_rel]} -> images imprinting "
              f"above 1e-12: {n_imprint_1e12}, above 1e-10: {n_imprint_1e10} (dense N' at T=1 is bounded by this, not by N)", flush=True)
        for first in a.first:
            for r in a.r:                                            # r INNERMOST (audit item 5)
                adaptable = list(range(first - 1, D))                # every module from the first adapted one, network order
                sels = layer_selections(a.layers, adaptable, a.maxL, seed, first)
                if not sels: continue
                adapted = sorted({l for _, _, ls in sels for l in ls})   # modules any pattern uses: A0, certificate, T arm
                gA = torch.Generator().manual_seed(seed + 11 + first)
                A0 = {}
                for l in adaptable:                                  # drawn for EVERY adaptable module in network order
                    A0_l = sig[l] * torch.randn(r, geo[l]["p_l"], generator=gA)
                    if l in adapted: A0[l] = A0_l.to(dev)
                arms = {}

                def layer_row(arm_name, T, l, C, Np, Np10, A_l, extra):
                    sA = float(torch.linalg.svdvals(A_l)[0])
                    crank = abs_rank(C, sA)
                    res = torch.linalg.norm(C @ Hbase[l], dim=0) / (torch.linalg.norm(A_l @ Hbase[l], dim=0) + 1e-300)
                    meta = dict(n_prime=Np, cert_rank=crank, sigma_max_A=sA, rank_B_at_1e10=Np10,
                                cert_residual_median=float(res.median()), cert_residual_max=float(res.max()),
                                rank_B_capped_by_width=(None if T == 0 else bool(Np >= min(r, geo[l]["cout"]))))
                    emit(dict(part="CONVLAYER", arm=arm_name, T=T, lr=a.lr, r=r, first_adapted=first,
                              layers_adapted=[x + 1 for x in adapted], layer=l + 1, kind=geo[l]["kind"],
                              p_l=geo[l]["p_l"], P_l=geo[l]["P"], NP=a.N * geo[l]["P"], out_width=geo[l]["cout"],
                              rank_cap=min(r, geo[l]["p_l"]), n_prime=Np, rank_B_at_1e10=Np10, n_prime_zero_drift=nprime_zd[l],
                              cert_rank=crank, cert_rank_formula=max(0, min(r, geo[l]["p_l"]) - Np),
                              conditions_per_image=crank * geo[l]["P"], vacuous=bool(crank == 0),
                              conv_vacuous=bool(nprime_zd[l] >= min(r, geo[l]["p_l"])),
                              rank_B_capped_by_width=meta["rank_B_capped_by_width"],
                              cert_residual_median=meta["cert_residual_median"], cert_residual_max=meta["cert_residual_max"],
                              certificate_holds_at_truth=bool(meta["cert_residual_median"] < 1e-8),
                              is_first_adapted=bool(l == adapted[0]), sigma0=sig[l], N=a.N, base=BASE,
                              git=PROV["git"], script_sha=PROV["script_sha"], **extra))
                    return meta

                Cs, meta = {}, {}
                for l in adapted:
                    Cs[l] = zero_drift_cert(A0[l], Ubase[l])
                    meta[l] = layer_row("zero_drift", 0, l, Cs[l], nprime_zd[l], nprime_zd[l], A0[l],
                                        dict(dense_invariant_ok=(bool(nprime_zd[l] == a.N) if geo[l]["kind"] != "conv" else None)))
                    print(f"  [r={r} first={first} zero_drift] module {l+1} ({geo[l]['kind']}): p_l={geo[l]['p_l']} N'={nprime_zd[l]} "
                          f"rank C={meta[l]['cert_rank']} (formula {max(0, min(r, geo[l]['p_l']) - nprime_zd[l])}) x P={geo[l]['P']}"
                          f"{'  [VACUOUS]' if meta[l]['cert_rank'] == 0 else ''}  residual {meta[l]['cert_residual_median']:.2e}", flush=True)
                arms["zero_drift"] = dict(Cs=Cs, meta=meta, T=0)
                for T in a.T_arm:
                    A0s = [A0[l] if l in A0 else None for l in range(D)]
                    t0 = time.time()
                    As, Bs = run_training6(X_img, Wms, bs_, Whead, bhead, Wd, bd, A0s, y, m, T, a.lr)
                    secs = time.time() - t0
                    finite = all(bool(torch.isfinite(As[l]).all() and torch.isfinite(Bs[l]).all()) for l in adapted)
                    if finite:
                        with torch.no_grad():
                            zT = forward6(X_img, Wms, bs_, Whead, bhead, Wd, bd, As, Bs)
                        lossT = float(F.cross_entropy(zT, y)); accT = float((zT.argmax(1) == y).double().mean())
                    else:
                        lossT = accT = float("nan")
                    diverged = (not finite) or not math.isfinite(lossT)
                    CsT, metaT = {}, {}
                    for l in adapted:
                        C, Np, S = certificate(As[l], Bs[l])
                        Np10 = int((S > 1e-10 * S[0]).sum()) if float(S[0]) > 0 else 0
                        is_first = (l == adapted[0]); is_dense = geo[l]["kind"] != "conv"
                        expect_first = (a.N if is_dense else min(nprime_zd[l], r, geo[l]["cout"])) if is_first else None
                        rel_zd = float(torch.linalg.norm(C - Cs[l]) / (torch.linalg.norm(A0[l]) + 1e-300)) if is_first else None
                        CsT[l] = C
                        metaT[l] = layer_row(f"T{T}", T, l, C, Np, Np10, As[l],
                                             dict(first_layer_expected_n_prime=expect_first,
                                                  first_layer_invariant_ok=(bool(Np == expect_first) if is_first else None),
                                                  dense_invariant_ok=(bool(Np == a.N) if (is_first and is_dense) else None),
                                                  n_prime_moves_vs_zero_drift=bool(Np != min(nprime_zd[l], r, geo[l]["cout"])),
                                                  base_softmax_residual_rel_per_image=imprint_rel,
                                                  n_images_imprinting_above_1e12=n_imprint_1e12,
                                                  n_images_imprinting_above_1e10=n_imprint_1e10,
                                                  dense_n_prime_matches_imprint_count=(bool(Np == n_imprint_1e12) if is_dense else None),
                                                  first_layer_cert_rel_diff_vs_zero_drift=rel_zd,
                                                  loss_before=loss0, loss_after=lossT, batch_acc_before=acc0, batch_acc_after=accT,
                                                  diverged=diverged, train_seconds=secs))
                        print(f"  [r={r} first={first} T={T}] module {l+1} ({geo[l]['kind']}): rank B_T={Np} (zd N'={nprime_zd[l]}, "
                              f"cap {min(r, geo[l]['cout'])}) rank C={metaT[l]['cert_rank']}{'  [VACUOUS]' if metaT[l]['cert_rank'] == 0 else ''} "
                              f"residual {metaT[l]['cert_residual_median']:.2e}"
                              + (f"  FIRST expected N'={expect_first} ok={Np == expect_first} |C_T-C_zd|/|A0|={rel_zd:.1e}" if is_first else "")
                              + (f"  imprinting images(1e-12)={n_imprint_1e12}" if is_dense else "")
                              + (f"  loss {loss0:.3e}->{lossT:.3e} acc {accT:.2f} diverged={diverged} {secs:.0f}s" if l == adapted[-1] else ""),
                              flush=True)
                    arms[f"T{T}"] = dict(Cs=CsT, meta=metaT, T=T, diverged=diverged, loss_after=lossT)
                CONF[(r, first)] = dict(adapted=adapted, sels=sels, arms=arms)
        SEEDS[seed] = dict(X_real=X_real, Hbase=Hbase, H0_norm=H0_norm, underflow=underflow, nprime_zd=nprime_zd, CONF=CONF)

    # ---- the chart sweep: one patch Jacobian per (chart, image); everything else is linear algebra ------------
    big = PCAChart(Xtr_t[:a.n_fit], npix, dev) if any(k < npix for k in a.ks) else None
    p_sizes = [gg["p_l"] * gg["P"] for gg in geo]
    for k in a.ks:
        tk = time.time()
        for seed in a.seed:
            S = SEEDS[seed]
            X_real, Hbase, H0_norm, underflow, nprime_zd, CONF = (S["X_real"], S["Hbase"], S["H0_norm"], S["underflow"],
                                                                  S["nprime_zd"], S["CONF"])
            if k >= npix:
                k_eff, chart_label, in_dim = npix, "pixel", npix
                X_domain = X_real; to_pixels = (lambda v: v); chart_err = 0.0
            else:
                chart = SubChart(big, k)
                k_eff, chart_label, in_dim = k, f"pca{k}", k
                X_domain = chart.coords_of(X_real)
                to_pixels = (lambda w, _c=chart: _c.psi(w[:, None])[:, 0])
                X_rec = chart.psi(X_domain)
                chart_err = float((torch.linalg.norm(X_rec - X_real, dim=0) / torch.linalg.norm(X_real, dim=0)).median())

            def flat_fn(v):
                return torch.cat([o.reshape(-1) for o in patch_maps(to_pixels(v), Wms, bs_, Wd, bd, geo, idxs)])

            # accumulators: d per (layer, image); per (conf, arm): q per (layer, image), ladder rows per (L, image)
            d_img = [[] for _ in range(D)]
            acc = {}
            for i in range(a.N):
                J = jac_patches(flat_fn, X_domain[:, i].contiguous(), a.jac_chunk)
                Jl = list(torch.split(J, p_sizes, dim=0)); del J
                for l in range(D):
                    d_img[l].append(numrank(torch.linalg.svdvals(Jl[l]), RANK_TOL))
                for (r, first), conf in CONF.items():
                    adapted, sels = conf["adapted"], conf["sels"]
                    for arm_name, arm in conf["arms"].items():
                        Cs, meta = arm["Cs"], arm["meta"]
                        st = acc.setdefault((r, first, arm_name), dict(q={l: [] for l in adapted},
                                                                       rows={si: [] for si in range(len(sels))},
                                                                       sv0={}))
                        blocks = {}
                        for l in adapted:
                            if meta[l]["cert_rank"] == 0:
                                st["q"][l].append(0); continue
                            blocks[l] = apply_cert(Cs[l], Jl[l], geo[l]["p_l"], geo[l]["P"])
                            st["q"][l].append(numrank(torch.linalg.svdvals(blocks[l]), RANK_TOL))
                        for si, (_, _, layers) in enumerate(sels):
                            parts = [blocks[l] for l in layers if l in blocks]
                            if not parts:
                                st["rows"][si].append(dict(sv=None, n_rows=0)); continue
                            S = torch.cat(parts, 0)
                            st["rows"][si].append(dict(sv=torch.linalg.svdvals(S).cpu(), n_rows=int(S.shape[0])))
                        del blocks
                del Jl
                if dev.type == "cuda": torch.cuda.empty_cache()
            d_layer = [med(d_img[l]) for l in range(D)]
            print(f"  [{chart_label} k={k_eff} seed={seed}] d_j = rank M_l per module = {d_layer} (per image {d_img}) chart_err={chart_err:.2e} "
                  f"({time.time()-tk:.0f}s)", flush=True)

            for (r, first), conf in CONF.items():
                adapted, sels = conf["adapted"], conf["sels"]
                for arm_name, arm in conf["arms"].items():
                    meta = arm["meta"]; st = acc[(r, first, arm_name)]
                    q_layer = {l: med(st["q"][l]) for l in adapted}
                    for si, (pat, draw, layers) in enumerate(sels):
                        L = len(layers)
                        k1 = d_layer[layers[0]]                   # nesting ceiling = the SHALLOWEST chosen module's d
                        dl = [d_layer[l] for l in layers]; ql = [q_layer[l] for l in layers]
                        Sq = sum(ql); cum = [sum(ql[:j]) for j in range(L)]
                        t52 = min(k1, Sq); corrected = min([dl[j] + cum[j] for j in range(L)] + [Sq])
                        rows, gaps, spec0, svs = [], [], None, []
                        for i, o in enumerate(st["rows"][si]):
                            if o["sv"] is None:
                                rows.append({**{lab: 0 for lab in LADDER}, "sigma_max": 0.0, "n_rows": 0}); svs.append(None); continue
                            sv = o["sv"]; svs.append(sv)
                            row = {lab: numrank(sv, TOL[lab]) for lab in LADDER}
                            row["sigma_max"] = float(sv[0]); row["n_rows"] = o["n_rows"]; rows.append(row)
                            if 0 < corrected < len(sv): gaps.append(float(sv[corrected - 1] / sv[corrected]))
                            if i == 0 and float(sv[0]) > 0: spec0 = [float(sv[j] / sv[0]) for j in range(len(sv))]
                        meas = {lab: med([o[lab] for o in rows]) for lab in LADDER}
                        ladder_vals = [meas[lab] for lab in LADDER]; fine = LADDER[-1]
                        gap_at_corr = med(gaps) if gaps else None
                        cond = cond_fields(svs, [o["1e-10"] for o in rows], [o["fp16"] for o in rows], corrected)
                        n_rows = med([o["n_rows"] for o in rows])
                        dead = bool(n_rows > 0 and med([o["sigma_max"] for o in rows]) < 1e-25)   # collapsed phi (a stack with
                        # NO rows -- every module vacuous -- is not a dead Jacobian, it is no_gap_vacuous: nothing was stacked)
                        match_c = bool(abs(meas[fine] - corrected) <= 1); match_t = bool(abs(meas[fine] - t52) <= 1)
                        # verdict at the 1e-10 rung (the record's rung; the finest rung sits at ambient wherever
                        # there is no gap, so a finest-rung verdict reads "neither" while 1e-10 lands exactly)
                        match_c10 = bool(abs(meas["1e-10"] - corrected) <= 1); match_t10 = bool(abs(meas["1e-10"] - t52) <= 1)
                        outcome, verdict = rank_outcome(dead, gap_at_corr, match_c10, match_t10)
                        q_formula = [min(meta[l]["cert_rank"] * geo[l]["P"], d_layer[l]) for l in layers]
                        q_dense_style = [min(meta[l]["cert_rank"], d_layer[l]) for l in layers]
                        emit(dict(part="RANKLAW", arm=arm_name, T=arm["T"],
                                  config=dict(model=a.ckpt.split("/")[-1], spec=a.spec, chart=chart_label, k=k_eff, in_dim=in_dim,
                                              first_adapted=first, seed=seed, r=r, N=a.N, depth=D, lr=a.lr,
                                              sigma0_rule=("given" if a.sigma0 is not None else "1/sqrt(p_l)"), label=a.label),
                                  n_layers=L, layers_in_objective=[l + 1 for l in layers], layer_kind=[geo[l]["kind"] for l in layers],
                                  layer_pattern=pat, pattern_draw=draw, nesting_ceiling_layer=layers[0] + 1,
                                  layers_adapted=[l + 1 for l in adapted],
                                  p_l=[geo[l]["p_l"] for l in layers], P_l=[geo[l]["P"] for l in layers],
                                  NP=[a.N * geo[l]["P"] for l in layers], width_bound=[geo[l]["width_bound"] for l in layers],
                                  n_prime=[meta[l]["n_prime"] for l in layers], n_prime_zero_drift=[nprime_zd[l] for l in layers],
                                  cert_rank=[meta[l]["cert_rank"] for l in layers],
                                  vacuous=[bool(meta[l]["cert_rank"] == 0) for l in layers],
                                  conv_vacuous=[bool(nprime_zd[l] >= min(r, geo[l]["p_l"])) for l in layers],
                                  layers_dropped_vacuous=[l + 1 for l in layers if meta[l]["cert_rank"] == 0],
                                  rank_B_capped_by_width=[meta[l]["rank_B_capped_by_width"] for l in layers],
                                  cert_residual_median=[meta[l]["cert_residual_median"] for l in layers],
                                  d_j=dl, d_j_per_image=[d_img[l] for l in layers], q_l=ql, q_l_per_image=[st["q"][l] for l in layers],
                                  k1=k1, sum_q=Sq, q_l_formula=q_formula, q_l_formula_dense_style=q_dense_style,
                                  q_l_measured_matches_formula=bool(ql == q_formula),
                                  q_l_measured_matches_dense_style=bool(ql == q_dense_style),
                                  t52_pred=int(t52), corrected_pred=int(corrected),
                                  measured_rank_by_tol=meas, measured_at_1e10=meas["1e-10"], measured_at_finest=meas[fine],
                                  stack_rows=int(n_rows), ambient_maxrank=int(min(n_rows, in_dim)),
                                  ladder_spread=int(max(ladder_vals) - min(ladder_vals)),
                                  ladder_converged=bool(meas[LADDER[-1]] - meas[LADDER[-2]] == 0),
                                  gap_at_corrected=gap_at_corr,
                                  real_rank_at_corrected=bool(gap_at_corr is not None and gap_at_corr > 10),
                                  spectrum_window_img0=spec0, spectrum_window_start=0 if spec0 else None,
                                  discriminates=bool(corrected < t52),
                                  saturated_below_k1=bool(meas[fine] < k1), saturated_at_k1=bool(meas[fine] >= k1 - 1),
                                  matches_corrected_at_finest=match_c, matches_t52_at_finest=match_t,
                                  dead_jacobian=dead, rank_test_outcome=outcome, rank_verdict=verdict, rank_verdict_rung="1e-10",   # the THIRD outcome
                                  chart_error=chart_err, cert_route_nullity_finest=in_dim - meas[fine],
                                  **cond,                       # cond_at_1e10 / cond_at_corrected / cond_at_fp16 (+ recompute keys)
                                  H0_norm_per_layer=[H0_norm[l] for l in layers], H0_underflow=[underflow[l] for l in layers],
                                  sigma_max_median=med([o["sigma_max"] for o in rows]),
                                  diverged=arm.get("diverged", False), loss_after=arm.get("loss_after"),
                                  claim_class="theory rank-law test at the truth on a conv encoder (no solve, no attack)",
                                  note=("zero_drift arm: C_l = P_{col(A0 U_l)^perp} A0 on the base patch span; T arms: C_l = "
                                        "P_{row(B_T)^perp} A_T from the release, applied to the BASE inputs (attacker-computable); "
                                        "vacuous modules contribute no rows; q_l_formula = min(rank(C_l)*P_l, d_l) is T5.2's "
                                        "per-layer budget under weight sharing, q_l_formula_dense_style = min(rank(C_l), d_l) "
                                        "is the audit's count"
                                        + ("" if pat == "prefix" else "; NON-PREFIX layer pattern: the corrected law's nesting "
                                           "reading over the chosen modules is what is being tested (plan P4)")),
                                  base=BASE, git=PROV["git"], script_sha=PROV["script_sha"], host=socket.gethostname(), cmd=" ".join(sys.argv)))
                        print(f"  [{chart_label} k={k_eff} seed={seed} r={r} first={first} {arm_name} {pat}{'' if draw is None else '#' + str(draw)} "
                              f"L={L}] modules={[l+1 for l in layers]} "
                              f"d_j={dl} q_l={ql} certrk={[meta[l]['cert_rank'] for l in layers]} k1={k1} Sq={Sq} "
                              f"| T5.2={t52} corr={corrected} discr={corrected < t52} "
                              f"| MEASURED {'/'.join(str(meas[t]) for t in LADDER)} rows={n_rows} "
                              f"{'BELOW k1' if meas[fine] < k1 else 'AT k1'} conv={meas[LADDER[-1]] - meas[LADDER[-2]] == 0} "
                              f"gap={gap_at_corr if gap_at_corr is None else round(gap_at_corr, 2)} "
                              f"cond1e10={cond['cond_at_1e10'] if cond['cond_at_1e10'] is None else round(cond['cond_at_1e10'], 2)} "
                              f"outcome={outcome}/{verdict}", flush=True)
            del acc
            print(f"  [{chart_label} k={k_eff} seed={seed}] done in {time.time()-tk:.0f}s", flush=True)


if __name__ == "__main__":
    main()
