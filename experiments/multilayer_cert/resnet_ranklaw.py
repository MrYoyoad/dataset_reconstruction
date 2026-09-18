#!/usr/bin/env python3
"""Zero-drift conv rank law on a SECOND real architecture: ResNet-18 / CIFAR-10 (plan 2026-09-18 package P2(ii),
Audit item 7).  Port of `conv_encoder_ranklaw.py` (same row format, same 12-rung tolerance ladder, same gap /
condition-number diagnostics, same third outcome) onto a torchvision ResNet-18 trained to the base gate by
`experiments/exact_inversion/train_resnet_backbone.py`, with LoRA slots on the four 3x3 convs of ONE residual stage.
THEORY test at the truth: zero drift, no LoRA training, no solve, no attack.

THE STAGE.  The brief says "stage 2: 128 channels, 8x8 positions, p_l = 128*9 = 1152, P_l = 64".  With the 3x3
stride-1 stem the torchvision stages sit at layer1 64@32x32, layer2 128@16x16, layer3 256@8x8, layer4 512@4x4, so
the stage whose INPUT is the 128-channel map and whose convs run at 8x8 positions is torchvision `layer3` (the
brief's "stage 2" counts from layer1 = stage 0).  Its four adaptable 3x3 convs, in network order:
    l=1  layer3.0.conv1  input 128@16x16, stride 2  -> p_l = 1152, P_l = 8*8 = 64   (matches the brief's arithmetic)
    l=2  layer3.0.conv2  input 256@8x8              -> p_l = 2304, P_l = 64
    l=3  layer3.1.conv1  input 256@8x8 (block-2 input) -> p_l = 2304, P_l = 64
    l=4  layer3.1.conv2  input 256@8x8              -> p_l = 2304, P_l = 64
N*P_l = 512 at N = 8 for every conv (`--stage` selects another stage; the arithmetic is recomputed and printed).

WHAT THE CERTIFICATE IS ON, in a residual net.  The certificate of an adapted conv is the MLP construction on the
conv's INPUT PATCHES: C_l = P_{col(A_{l,0} U_l)^perp} A_{l,0}, U_l a basis of the span of the N*P_l base patch
vectors unfold(h_l^0) at the truths, rank min(r, p_l) - N'_l with N'_l the patch-span rank.  Those inputs are: the
block's input for conv1 (the previous stage's output for block 1, block 1's output for block 2) and conv1's
activation relu(bn1(conv1(.))) for conv2.  The SKIP path (identity or the 1x1 downsample) adds into the block
OUTPUT and never enters any adapted conv's input, so it does not enter the certificate; it enters the chart
Jacobian of the deeper convs' inputs like every other frozen op.  BatchNorm in eval mode is a fixed per-channel
affine map z -> s*z + t (s = gamma/sqrt(var+eps), t = beta - mean*s) and folds into the frozen map; the functional
forward below writes it as exactly that and is CHECKED at load against the module forward (FP64, < 1e-10).
ReLU (torchvision's) is piecewise linear, so the chart Jacobian is defined a.e. and forward-mode AD is exact.

PRE-REGISTERED live / vacuous set per conv (same rule as the CNN section: conv-vacuous iff N'_l >= min(r, p_l)).
    N'_l <= N*P_l = 512 < p_l for every conv of the stage, so the p_l-SATURATION death of the MNIST CNN's conv 1
    (N' = p_l = 9) CANNOT occur here; the only vacuity is the r-side one, N'_l >= r:
      r = 64  : live iff N'_l < 64   -- vacuous unless the 512 patch vectors span < 64 directions
      r = 256 : live iff N'_l < 256
      r = 512 : live iff N'_l < 512  -- live iff the 512 base patch vectors are linearly DEPENDENT at 1e-10
      r = 1024: live by arithmetic (rank C_l >= 512 whatever N'_l is); smoke code-path rank, not in the brief's set
    The measured N'_l decides; a run in which every (r, conv) is vacuous reports `no_gap_vacuous` on every stacked
    row (nothing stacked), never an integer.  Conditions per image where live: rank C_l * 64.
PRE-REGISTERED laws, over the stacked prefix of L convs (L = 1..4), as in the CNN section:
    T5.2 (weight sharing): min(k_1, sum_l q_l) with q_l = min(rank C_l * P_l, d_l);  corrected: min_j(d_j + sum_{l<j} q_l);
    d_l = rank of the chart Jacobian of vec(P_l(psi(v))) = rank of the Jacobian of the conv's INPUT REPRESENTATION
    (unfold with stride 1 pad 1, or stride 2 pad 1 on an even side, reads every input element at least once, so it
    is an injective linear map and the two ranks coincide; `--verify-patch-rank` measures both at small k and
    records `d_patch_equals_d_rep`).  Width bounds on d_l: min(k, 32768) for l = 1, min(k, 16384) for l = 2..4.
    Where a single live conv already gives q_l >= k the laws coincide (the CNN section's VACUOUS-for-discrimination
    outcome); where they differ the row's `discriminates` is true and the 1e-10 rung is read against both.
    No gap at the cut -> ladder and spectrum, never an integer (`rank_test_outcome` in {compare, no_gap_vacuous,
    dead}, imported from `real_encoder_ranklaw.rank_outcome`).
CHARTS.  PCA k in {32, 128, 384, 1024, 3072} fit on the CIFAR-100 TRAIN split (all 100 classes, 50 000 images: the
public pool) + the pixel chart (k >= 3072 = identity).  Truths: the ladder's eight CIFAR-100 motorcycles (the
motorcycle TEST split, randperm under seed+7, first N -- `experiments/oracle_ladder/ladder_cell.py`'s join key).
HOW THE STACKED JACOBIAN IS BUILT.  One forward-mode Jacobian per (chart, image) of the concatenated conv INPUT
representations (81 920 x k at this stage) with `torch.func.jacfwd`, chunked through vmap(jvp) by `--jac-chunk`
(`conv_encoder_ranklaw.jac_patches`); per conv the patch Jacobian is unfold applied to its columns, and the
certified block is (C_l (x) I_{P_l}) J_l with C_l replaced by its ROW-SPACE REDUCTION U_c^T C_l (rank C_l rows,
U_c orthonormal): C_l = U_c (U_c^T C_l) up to the numerically-zero part below the 1e-10 * sigma_max(A_0) floor, and a
left factor with orthonormal columns leaves singular values unchanged, so ranks, gaps and condition numbers are
those of the full stack at a fraction of the rows.  Memory at k = 3072: representation Jacobian 2.0 GB per image,
patch Jacobian <= 3.6 GB transient per conv, certified block <= 0.8 GB per conv at r = 512, stacked <= 3.2 GB,
forward-mode intermediates ~6-8 MB per tangent (chunk 256 -> ~2 GB): peak ~15-20 GB.  FP64 throughout.
WP0 on every row: the checkpoint's recorded train acc / CE / test acc AND the values re-measured at load on the
full splits in FP64; `fully_trained_gate` = train acc >= 0.995 and train CE <= 1e-2.

  python -u -m experiments.multilayer_cert.resnet_ranklaw --ckpt models/exact_inversion/cifar10_resnet18.pth \
      --r 64 256 512 --ks 32 128 384 1024 3072 --maxL 4 --jac-chunk 256
"""
import argparse, json, math, os, pickle, socket, sys, time
import numpy as np
import torch, torch.nn.functional as F

from experiments.exact_inversion.trained_backbone import PCAChart
from experiments.exact_inversion.train_resnet_backbone import load_resnet18, GATE_TRAIN_ACC, GATE_TRAIN_LOSS
from experiments.cifar.cifar_newclass import load_cifar10, load_cifar100_class
from experiments.multilayer_cert.common import provenance, numrank as numrank_abs
from experiments.multilayer_cert.real_encoder_ranklaw import (LADDER, TOL, med, numrank, zero_drift_cert,
                                                              cond_fields, rank_outcome)
from experiments.multilayer_cert.conv_encoder_ranklaw import SubChart, jac_patches, base_stats

torch.set_default_dtype(torch.float64)
RANK_TOL = 1e-10
NPIX = 3 * 32 * 32


def load_cifar100_train(root):
    """The public pool: the whole CIFAR-100 train split, (50000, 3072) in [0, 1], CIFAR layout."""
    b = pickle.load(open(os.path.join(root, "cifar-100-python", "train"), "rb"), encoding="bytes")
    return b[b"data"].astype(np.float64) / 255.0


# ----------------------------------------------------------------------------------------------------------------
# the frozen map, written functionally (BN eval = per-channel affine) so torch.func can trace it
def bn_st(bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return s[None, :, None, None], (bn.bias - bn.running_mean * s)[None, :, None, None]


def conv_bn(x, conv, bn):
    s, t = bn_st(bn)
    return F.conv2d(x, conv.weight, None, conv.stride, conv.padding) * s + t


def block_forward(h, blk, record=None):
    """BasicBlock: out = relu(bn2(conv2(relu(bn1(conv1(h))))) + skip(h)).  `record` collects the two conv INPUTS."""
    a = F.relu(conv_bn(h, blk.conv1, blk.bn1))
    if record is not None: record += [h, a]
    o = conv_bn(a, blk.conv2, blk.bn2)
    idn = conv_bn(h, blk.downsample[0], blk.downsample[1]) if blk.downsample is not None else h
    return F.relu(o + idn)


def stage_inputs(x, net, stage):
    """(B,3,32,32) -> (list of the adapted stage's conv INPUT maps in network order, the stage's output)."""
    h = F.relu(conv_bn(x, net.conv1, net.bn1))
    for name in ("layer1", "layer2", "layer3", "layer4"):
        rec = [] if name == stage else None
        for blk in getattr(net, name):
            h = block_forward(h, blk, rec)
        if name == stage: return rec, h
    raise ValueError(stage)


def functional_logits(x, net):
    h = F.relu(conv_bn(x, net.conv1, net.bn1))
    for name in ("layer1", "layer2", "layer3", "layer4"):
        for blk in getattr(net, name): h = block_forward(h, blk)
    return F.linear(h.mean((2, 3)), net.fc.weight, net.fc.bias)


def stage_geometry(net, stage, ins):
    """One entry per adaptable conv of the stage, from the module attributes and the probe inputs."""
    geo = []
    for b, blk in enumerate(getattr(net, stage)):
        for which, conv in (("conv1", blk.conv1), ("conv2", blk.conv2)):
            l = len(geo); C, H, W = ins[l].shape[1:]
            k, s, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]
            hout = (H + 2 * p - k) // s + 1
            geo.append(dict(name=f"{stage}.{b}.{which}", block=b + 1, which=which, kind="conv", cin=C, cout=conv.out_channels,
                            k=k, s=s, p=p, hin=H, hout=hout, p_l=C * k * k, P=hout * hout, in_rep=C * H * W,
                            out_dim=conv.out_channels * hout * hout))
    for l, g in enumerate(geo): g["width_bound"] = min(g2["in_rep"] for g2 in geo[:l + 1])
    return geo


def unfold_rep(Jl, g, kdim):
    """Columns of the representation Jacobian (rep, k) -> patch Jacobian (k, p_l, P_l) by the same unfold."""
    return F.unfold(Jl.T.reshape(kdim, g["cin"], g["hin"], g["hin"]), g["k"], padding=g["p"], stride=g["s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/exact_inversion/cifar10_resnet18.pth")
    ap.add_argument("--stage", default="layer3", choices=["layer1", "layer2", "layer3", "layer4"])
    ap.add_argument("--cls", default="motorcycle", help="CIFAR-100 class of the private images (the ladder's eight)")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", nargs="*", type=int, default=[64, 256, 512])
    ap.add_argument("--ks", nargs="*", type=int, default=[32, 128, 384, 1024, 3072], help="k >= 3072 is the pixel chart")
    ap.add_argument("--maxL", type=int, default=4, help="prefix depths L = 1..maxL over the stage's convs")
    ap.add_argument("--seed", nargs="*", type=int, default=[1])
    ap.add_argument("--sigma0", type=float, default=None, help="A_0 scale; default per conv 1/sqrt(p_l)")
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--jac-chunk", type=int, default=256)
    ap.add_argument("--verify-patch-rank", action="store_true", help="also rank the patch Jacobian at k <= 128")
    ap.add_argument("--gate-chunk", type=int, default=500)
    ap.add_argument("--label", default="")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    PROV = provenance(__file__); first = 1

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    # ---- backbone, WP0 numbers at load, functional-forward check --------------------------------------------
    net, ck = load_resnet18(a.ckpt, dev)
    Xtr, ytr, Xte, yte = load_cifar10(a.data_root)
    Xtr_t = torch.tensor(Xtr, device=dev).double().reshape(-1, 3, 32, 32); ytr_t = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev).double().reshape(-1, 3, 32, 32); yte_t = torch.tensor(yte, device=dev)
    t0 = time.time()
    tr_acc, tr_loss = base_stats(Xtr_t, ytr_t, net, chunk=a.gate_chunk); te_acc, te_loss = base_stats(Xte_t, yte_t, net, chunk=a.gate_chunk)
    BASE = dict(ckpt=a.ckpt, arch=ck.get("arch"), ckpt_keys=sorted(k for k in ck if k != "history"),
                ckpt_test_acc=ck.get("test_acc"), ckpt_train_acc=ck.get("train_acc"), ckpt_train_loss=ck.get("train_loss"),
                ckpt_epochs=ck.get("epochs_run"), ckpt_rule_met=ck.get("rule_met"), ckpt_git=ck.get("git"),
                measured_train_acc=tr_acc, measured_train_loss=tr_loss, measured_test_acc=te_acc, measured_test_loss=te_loss,
                fully_trained_gate=bool(tr_acc >= GATE_TRAIN_ACC and tr_loss <= GATE_TRAIN_LOSS),
                gate=f"train acc >= {GATE_TRAIN_ACC} and train loss <= {GATE_TRAIN_LOSS} (WP0)", label=a.label, stage=a.stage)
    print(f"# resnet_ranklaw ckpt={a.ckpt} stage={a.stage} cls={a.cls} N={a.N} r={a.r} ks={a.ks} maxL={a.maxL} seeds={a.seed} "
          f"jac_chunk={a.jac_chunk} label={a.label!r} git={PROV['git']} script_sha={PROV['script_sha']} dev={dev}", flush=True)
    print(f"# BASE (WP0): ckpt train_acc={BASE['ckpt_train_acc']} train_loss={BASE['ckpt_train_loss']} test_acc={BASE['ckpt_test_acc']} "
          f"epochs={BASE['ckpt_epochs']} rule_met={BASE['ckpt_rule_met']} | MEASURED (FP64, BN eval) train acc {tr_acc*100:.3f}% "
          f"CE {tr_loss:.3e} test acc {te_acc*100:.2f}% CE {te_loss:.3e} | fully_trained_gate={BASE['fully_trained_gate']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    print("# THEORY TEST of the zero-drift conv rank law on a ResNet stage, NOT an attack config", flush=True)
    with torch.no_grad():                                        # BN-as-affine functional forward == module forward
        xb = Xte_t[:64]
        fdiff = float((functional_logits(xb, net) - net(xb)).abs().max())
    print(f"# functional forward (BN eval as per-channel affine) vs module forward: max abs diff {fdiff:.1e}", flush=True)
    assert fdiff < 1e-9, f"functional forward disagrees with the module forward: {fdiff}"

    # ---- public pool (CIFAR-100 train) and private images (the ladder's eight motorcycles) -----------------------
    Pub = torch.tensor(load_cifar100_train(a.data_root)[:a.n_fit], device=dev)
    pool, cname = load_cifar100_class(a.data_root, a.cls)
    Pri = torch.tensor(pool["test"], dtype=torch.float64, device=dev)
    big = PCAChart(Pub, NPIX, dev) if any(k < NPIX for k in a.ks) else None
    print(f"# public pool: CIFAR-100 train {tuple(Pub.shape)}; private pool: CIFAR-100 '{cname}' test {tuple(Pri.shape)}", flush=True)

    SEEDS = {}
    for seed in a.seed:
        g = torch.Generator().manual_seed(seed + 7)
        perm = torch.randperm(Pri.shape[0], generator=g)
        idx = [int(v) for v in perm[:a.N]]
        X_real = Pri[idx].T.contiguous()                             # (3072, N)
        X_img = X_real.T.reshape(a.N, 3, 32, 32).contiguous()
        with torch.no_grad():
            ins, _ = stage_inputs(X_img, net, a.stage)
            pred = net(X_img).argmax(1).tolist()
        geo = stage_geometry(net, a.stage, ins); D = len(geo)
        print(f"# private join-key indices into the {cname} test split (seed {seed}+7, first {a.N}): {idx}; base CIFAR-10 argmax {pred}", flush=True)
        # base patch matrices, spans, certificates
        Hbase, Ubase, nprime = [], [], []
        for l, gg in enumerate(geo):
            Pt = F.unfold(ins[l], gg["k"], padding=gg["p"], stride=gg["s"])          # (N, p_l, P_l)
            assert Pt.shape[1] == gg["p_l"] and Pt.shape[2] == gg["P"]
            Hl = Pt.permute(1, 0, 2).reshape(gg["p_l"], -1).contiguous(); Hbase.append(Hl)
            U, S, _ = torch.linalg.svd(Hl, full_matrices=False)
            rk = numrank(S, RANK_TOL); Ubase.append(U[:, :rk].contiguous()); nprime.append(rk)
            emit(dict(part="PATCH_SPAN", seed=seed, layer=l + 1, name=gg["name"], kind="conv", p_l=gg["p_l"], P_l=gg["P"],
                      n_patch_vectors=a.N * gg["P"], patch_span_rank=rk, width_bound=gg["width_bound"], in_rep=gg["in_rep"],
                      spans_input_dim=bool(rk >= gg["p_l"]), patch_vectors_dependent=bool(rk < a.N * gg["P"]),
                      conv_vacuous_at_r={str(r): bool(rk >= min(r, gg["p_l"])) for r in a.r},
                      cert_rank_formula_at_r={str(r): max(0, min(r, gg["p_l"]) - rk) for r in a.r},
                      H0_norm=float(torch.linalg.norm(Hl, dim=0).median()), sigma_ratio_last=float(S[rk - 1] / S[0]) if rk else None,
                      sigma_ratio_next=float(S[rk] / S[0]) if rk < len(S) else None,
                      N=a.N, private_join_idx=idx, base=BASE, git=PROV["git"], script_sha=PROV["script_sha"]))
            print(f"  conv {l+1} {gg['name']}: input {gg['cin']}@{gg['hin']}x{gg['hin']} s{gg['s']} -> p_l={gg['p_l']} P_l={gg['P']} "
                  f"N*P={a.N*gg['P']} in_rep={gg['in_rep']} width_bound={gg['width_bound']} patch span rank N'={rk} "
                  f"(sigma_N'/sigma_1 {float(S[rk-1]/S[0]) if rk else 0:.1e}, next {float(S[rk]/S[0]) if rk < len(S) else 0:.1e})", flush=True)
        print("# LIVE/VACUOUS SET from the measured spans (conv-vacuous iff N' >= min(r, p_l)):", flush=True)
        for r in a.r:
            cr = [max(0, min(r, gg["p_l"]) - nprime[l]) for l, gg in enumerate(geo)]
            print(f"#   r={r}: rank C_l = {cr}; vacuous convs {[l+1 for l in range(D) if cr[l] == 0]}; conditions/image {[c*geo[l]['P'] for l, c in enumerate(cr)]}", flush=True)
            for k in a.ks:
                k_eff = min(k, NPIX); dpred = [min(k_eff, gg["width_bound"]) for gg in geo]
                qpred = [min(cr[l] * geo[l]["P"], dpred[l]) for l in range(D)]
                for L in range(1, min(D, a.maxL) + 1):
                    Sq = sum(qpred[:L]); t52 = min(dpred[0], Sq); corr = min([dpred[j] + sum(qpred[:j]) for j in range(L)] + [Sq])
                    print(f"#   r={r} k={k_eff} L={L}: d_pred={dpred[:L]} q_pred={qpred[:L]} T5.2={t52} corrected={corr} discriminates={corr < t52}", flush=True)
        sig = [a.sigma0 if a.sigma0 is not None else 1.0 / math.sqrt(gg["p_l"]) for gg in geo]
        CONF = {}
        for r in a.r:
            gA = torch.Generator().manual_seed(seed + 11 + first)
            A0 = [(sig[l] * torch.randn(r, geo[l]["p_l"], generator=gA)).to(dev) for l in range(D)]
            Cs, Cred, meta = [], [], []
            for l in range(D):
                C = zero_drift_cert(A0[l], Ubase[l]); sA = float(torch.linalg.svdvals(A0[l])[0])
                crank, svC = numrank_abs(C, RANK_TOL, ref=sA)
                res = torch.linalg.norm(C @ Hbase[l], dim=0) / (torch.linalg.norm(A0[l] @ Hbase[l], dim=0) + 1e-300)
                if crank > 0:
                    Uc, _, _ = torch.linalg.svd(C, full_matrices=False); Cr = (Uc[:, :crank].T @ C).contiguous()
                else:
                    Cr = None
                m = dict(cert_rank=crank, cert_rank_formula=max(0, min(r, geo[l]["p_l"]) - nprime[l]), sigma_max_A=sA,
                         cert_residual_median=float(res.median()), cert_residual_max=float(res.max()),
                         reduced_rows=(0 if Cr is None else int(Cr.shape[0])))
                Cs.append(C); Cred.append(Cr); meta.append(m)
                emit(dict(part="CONVLAYER", arm="zero_drift", T=0, seed=seed, r=r, first_adapted=first, layers_adapted=list(range(1, D + 1)),
                          layer=l + 1, name=geo[l]["name"], kind="conv", p_l=geo[l]["p_l"], P_l=geo[l]["P"], NP=a.N * geo[l]["P"],
                          out_width=geo[l]["cout"], rank_cap=min(r, geo[l]["p_l"]), n_prime=nprime[l], cert_rank=crank,
                          cert_rank_formula=m["cert_rank_formula"], cert_rank_matches_formula=bool(crank == m["cert_rank_formula"]),
                          conditions_per_image=crank * geo[l]["P"], vacuous=bool(crank == 0),
                          conv_vacuous=bool(nprime[l] >= min(r, geo[l]["p_l"])), sigma_max_A=sA,
                          cert_residual_median=m["cert_residual_median"], cert_residual_max=m["cert_residual_max"],
                          certificate_holds_at_truth=bool(m["cert_residual_median"] < 1e-8), sigma0=sig[l], N=a.N,
                          base=BASE, git=PROV["git"], script_sha=PROV["script_sha"]))
                print(f"  [r={r} zero_drift] conv {l+1} {geo[l]['name']}: p_l={geo[l]['p_l']} N'={nprime[l]} rank C={crank} "
                      f"(formula {m['cert_rank_formula']}) x P={geo[l]['P']}{'  [VACUOUS]' if crank == 0 else ''}  "
                      f"residual med {m['cert_residual_median']:.2e} max {m['cert_residual_max']:.2e}", flush=True)
            CONF[r] = dict(Cred=Cred, meta=meta)
            del Cs
        SEEDS[seed] = dict(X_real=X_real, geo=geo, nprime=nprime, Hbase=Hbase, CONF=CONF, idx=idx)

    # ---- the chart sweep -------------------------------------------------------------------------------------
    for k in a.ks:
        tk = time.time()
        for seed in a.seed:
            S_ = SEEDS[seed]; X_real, geo, nprime, CONF = S_["X_real"], S_["geo"], S_["nprime"], S_["CONF"]; D = len(geo)
            if k >= NPIX:
                k_eff, chart_label, in_dim = NPIX, "pixel", NPIX
                X_domain = X_real; to_pixels = (lambda v: v); chart_err = 0.0
            else:
                chart = SubChart(big, k); k_eff, chart_label, in_dim = k, f"pca{k}", k
                X_domain = chart.coords_of(X_real); to_pixels = (lambda w, _c=chart: _c.psi(w[:, None])[:, 0])
                chart_err = float((torch.linalg.norm(chart.psi(X_domain) - X_real, dim=0) / torch.linalg.norm(X_real, dim=0)).median())
            rep_sizes = [gg["in_rep"] for gg in geo]

            def flat_fn(v):
                ins, _ = stage_inputs(to_pixels(v).reshape(1, 3, 32, 32), net, a.stage)
                return torch.cat([h.reshape(-1) for h in ins])

            d_img = [[] for _ in range(D)]; dpatch_img = [[] for _ in range(D)]
            acc = {r: dict(q=[[] for _ in range(D)], rows=[[] for _ in range(min(D, a.maxL))]) for r in a.r}
            for i in range(a.N):
                ti = time.time()
                J = jac_patches(flat_fn, X_domain[:, i].contiguous(), a.jac_chunk)       # (sum rep, k)
                Jl = list(torch.split(J, rep_sizes, dim=0)); del J
                for l in range(D):
                    d_img[l].append(numrank(torch.linalg.svdvals(Jl[l]), RANK_TOL))
                    if a.verify_patch_rank and k_eff <= 128:
                        Jp = unfold_rep(Jl[l], geo[l], k_eff).reshape(k_eff, -1).T
                        dpatch_img[l].append(numrank(torch.linalg.svdvals(Jp), RANK_TOL)); del Jp
                for r in a.r:
                    Cred, meta = CONF[r]["Cred"], CONF[r]["meta"]; blocks = {}
                    for l in range(D):
                        if Cred[l] is None:
                            acc[r]["q"][l].append(0); continue
                        Jp = unfold_rep(Jl[l], geo[l], k_eff)                                   # (k, p_l, P_l)
                        blk = torch.matmul(Cred[l], Jp).permute(1, 2, 0).reshape(-1, k_eff).contiguous(); del Jp
                        acc[r]["q"][l].append(numrank(torch.linalg.svdvals(blk), RANK_TOL)); blocks[l] = blk
                    for L in range(1, min(D, a.maxL) + 1):
                        parts = [blocks[l] for l in range(L) if l in blocks]
                        if not parts:
                            acc[r]["rows"][L - 1].append(dict(sv=None, n_rows=0)); continue
                        Sm = torch.cat(parts, 0)
                        acc[r]["rows"][L - 1].append(dict(sv=torch.linalg.svdvals(Sm).cpu(), n_rows=int(Sm.shape[0]))); del Sm
                    del blocks
                del Jl
                if dev.type == "cuda": torch.cuda.empty_cache()
                q_str = "; ".join("r=%d: %s" % (r, [acc[r]["q"][l][-1] for l in range(D)]) for r in a.r)
                d_str = ("  d_patch=%s" % [dpatch_img[l][-1] for l in range(D)]) if dpatch_img[0] else ""
                print("  [%s k=%d seed=%d] image %d: d_j=%s%s  q_l %s (%.0fs)" % (
                    chart_label, k_eff, seed, i, [d_img[l][-1] for l in range(D)], d_str, q_str, time.time() - ti), flush=True)
            d_layer = [med(d_img[l]) for l in range(D)]
            print(f"  [{chart_label} k={k_eff} seed={seed}] d_j = {d_layer} (per image {d_img}) chart_err={chart_err:.2e} ({time.time()-tk:.0f}s)", flush=True)
            for r in a.r:
                meta = CONF[r]["meta"]; st = acc[r]; q_layer = [med(st["q"][l]) for l in range(D)]
                for L in range(1, min(D, a.maxL) + 1):
                    layers = list(range(L)); k1 = d_layer[0]
                    dl = [d_layer[l] for l in layers]; ql = [q_layer[l] for l in layers]
                    Sq = sum(ql); cum = [sum(ql[:j]) for j in range(L)]
                    t52 = min(k1, Sq); corrected = min([dl[j] + cum[j] for j in range(L)] + [Sq])
                    rows, gaps, spec0, svs = [], [], None, []
                    for i, o in enumerate(st["rows"][L - 1]):
                        if o["sv"] is None:
                            rows.append({**{lab: 0 for lab in LADDER}, "sigma_max": 0.0, "n_rows": 0}); svs.append(None); continue
                        sv = o["sv"]; svs.append(sv)
                        row = {lab: numrank(sv, TOL[lab]) for lab in LADDER}
                        row["sigma_max"] = float(sv[0]); row["n_rows"] = o["n_rows"]; rows.append(row)
                        if 0 < corrected < len(sv): gaps.append(float(sv[corrected - 1] / sv[corrected]))
                        if i == 0 and float(sv[0]) > 0: spec0 = [float(sv[j] / sv[0]) for j in range(min(len(sv), 4096))]
                    meas = {lab: med([o[lab] for o in rows]) for lab in LADDER}
                    ladder_vals = [meas[lab] for lab in LADDER]; fine = LADDER[-1]
                    gap_at_corr = med(gaps) if gaps else None
                    cond = cond_fields(svs, [o["1e-10"] for o in rows], [o["fp16"] for o in rows], corrected)
                    n_rows = med([o["n_rows"] for o in rows])
                    dead = bool(n_rows > 0 and med([o["sigma_max"] for o in rows]) < 1e-25)
                    match_c = bool(abs(meas[fine] - corrected) <= 1); match_t = bool(abs(meas[fine] - t52) <= 1)
                    match_c10 = bool(abs(meas["1e-10"] - corrected) <= 1); match_t10 = bool(abs(meas["1e-10"] - t52) <= 1)
                    outcome, verdict = rank_outcome(dead, gap_at_corr, match_c10, match_t10)
                    q_formula = [min(meta[l]["cert_rank"] * geo[l]["P"], d_layer[l]) for l in layers]
                    q_dense_style = [min(meta[l]["cert_rank"], d_layer[l]) for l in layers]
                    emit(dict(part="RANKLAW", arm="zero_drift", T=0,
                              config=dict(model=a.ckpt.split("/")[-1], arch="resnet18_cifar", stage=a.stage, chart=chart_label, k=k_eff,
                                          in_dim=in_dim, first_adapted=first, seed=seed, r=r, N=a.N, depth=D, cls=cname,
                                          sigma0_rule=("given" if a.sigma0 is not None else "1/sqrt(p_l)"), label=a.label),
                              n_layers=L, layers_in_objective=[l + 1 for l in layers], layer_names=[geo[l]["name"] for l in layers],
                              layer_kind=["conv"] * L, layer_pattern="prefix", pattern_draw=None, nesting_ceiling_layer=1,
                              layers_adapted=list(range(1, D + 1)),
                              p_l=[geo[l]["p_l"] for l in layers], P_l=[geo[l]["P"] for l in layers], NP=[a.N * geo[l]["P"] for l in layers],
                              width_bound=[geo[l]["width_bound"] for l in layers],
                              n_prime=[nprime[l] for l in layers], n_prime_zero_drift=[nprime[l] for l in layers],
                              cert_rank=[meta[l]["cert_rank"] for l in layers], vacuous=[bool(meta[l]["cert_rank"] == 0) for l in layers],
                              conv_vacuous=[bool(nprime[l] >= min(r, geo[l]["p_l"])) for l in layers],
                              layers_dropped_vacuous=[l + 1 for l in layers if meta[l]["cert_rank"] == 0],
                              cert_residual_median=[meta[l]["cert_residual_median"] for l in layers],
                              d_j=dl, d_j_per_image=[d_img[l] for l in layers],
                              d_patch_per_image=([dpatch_img[l] for l in layers] if dpatch_img[0] else None),
                              d_patch_equals_d_rep=(bool(all(dpatch_img[l] == d_img[l] for l in layers)) if dpatch_img[0] else None),
                              q_l=ql, q_l_per_image=[st["q"][l] for l in layers],
                              k1=k1, sum_q=Sq, q_l_formula=q_formula, q_l_formula_dense_style=q_dense_style,
                              q_l_measured_matches_formula=bool(ql == q_formula), q_l_measured_matches_dense_style=bool(ql == q_dense_style),
                              t52_pred=int(t52), corrected_pred=int(corrected),
                              measured_rank_by_tol=meas, measured_at_1e10=meas["1e-10"], measured_at_finest=meas[fine],
                              stack_rows=int(n_rows), reduced_certificate_rows=[meta[l]["reduced_rows"] for l in layers],
                              ambient_maxrank=int(min(n_rows, in_dim)),
                              ladder_spread=int(max(ladder_vals) - min(ladder_vals)),
                              ladder_converged=bool(meas[LADDER[-1]] - meas[LADDER[-2]] == 0),
                              gap_at_corrected=gap_at_corr, real_rank_at_corrected=bool(gap_at_corr is not None and gap_at_corr > 10),
                              spectrum_window_img0=spec0, spectrum_window_start=0 if spec0 else None,
                              discriminates=bool(corrected < t52),
                              saturated_below_k1=bool(meas[fine] < k1), saturated_at_k1=bool(meas[fine] >= k1 - 1),
                              matches_corrected_at_finest=match_c, matches_t52_at_finest=match_t,
                              dead_jacobian=dead, rank_test_outcome=outcome, rank_verdict=verdict, rank_verdict_rung="1e-10",
                              chart_error=chart_err, cert_route_nullity_finest=in_dim - meas[fine],
                              **cond,
                              sigma_max_median=med([o["sigma_max"] for o in rows]),
                              claim_class="theory rank-law test at the truth on a ResNet-18 stage (zero drift, no solve, no attack)",
                              note=("zero_drift: C_l = P_{col(A0 U_l)^perp} A0 on the base patch span of each adapted conv's INPUT "
                                    "(skip path not in the certificate; BN eval folded as affine); stacked rows use the row-space "
                                    "reduction U_c^T C_l (identical singular values); q_l_formula = min(rank(C_l)*P_l, d_l) is T5.2's "
                                    "budget under weight sharing; d_l = rank of the conv-input-representation Jacobian "
                                    "(= patch-Jacobian rank, unfold injective)"),
                              base=BASE, git=PROV["git"], script_sha=PROV["script_sha"], host=socket.gethostname(), cmd=" ".join(sys.argv)))
                    print(f"  [{chart_label} k={k_eff} seed={seed} r={r} L={L}] convs={[l+1 for l in layers]} d_j={dl} q_l={ql} "
                          f"certrk={[meta[l]['cert_rank'] for l in layers]} k1={k1} Sq={Sq} | T5.2={t52} corr={corrected} discr={corrected < t52} "
                          f"| MEASURED {'/'.join(str(meas[t]) for t in LADDER)} rows={n_rows} "
                          f"gap={gap_at_corr if gap_at_corr is None else round(gap_at_corr, 2)} "
                          f"cond1e10={cond['cond_at_1e10'] if cond['cond_at_1e10'] is None else round(cond['cond_at_1e10'], 2)} "
                          f"outcome={outcome}/{verdict}", flush=True)
            del acc
            print(f"  [{chart_label} k={k_eff} seed={seed}] done in {time.time()-tk:.0f}s", flush=True)


if __name__ == "__main__":
    main()
