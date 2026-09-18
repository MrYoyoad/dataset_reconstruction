#!/usr/bin/env python3
"""P3 / P6 harness: LoRA on a CHOSEN SUBSET of layers of a checkpoint-loaded MLP, the drift of an adapted layer's
input measured at the truth, both certificates at the truth, and the per-image certificate LM solve.  FP64.

Plan: notes/plan_2026-09-18_multilayer_parameter_program.md, packages P3 and P6 as amended by its "Audit 2026-09-18"
items 1-4.  The audit found no harness that (a) loads a trained checkpoint, (b) adapts a subset of layers (so the
input to a target layer DRIFTS because the layer below it trains), (c) trains with momentum / weight decay and
(d) solves.  survival.py has (b)-(c) on a random synthetic net without a solve; the ladder has (a)+(d) on the head
only.  This file generalises deep_stack.run_training_deep with an ADAPTER MASK (None = frozen) and replicates
survival.py's definitions exactly (see `layer_stats`), then runs the ladder's solver (certificate.lm_cert).

NETWORK.  deep_stack convention: Ws[0..D-1] weight matrices (1-indexed layers 1..D in the CLI), bias on layer 1
only, GELU between layers, none after the head.  `mnist_mlp_d15w1000_full.pth` (the d15 TWIN, passes the WP0 gate)
loads through load_deep; `mnist_mlp_strong.pth` (784-1000-1000-10, state_dict layout) through its own loader here.
PRIVATES.  The 18 Sept package's eight: EMNIST letter "a", TEST split, `randperm(seed+7)[:N]` -- the join key of
ladder_cell.py; `--private raw` (default) trains on the pixels, `--private onchart` on their PCA-k projections (the
truth is then IN the chart: tier 1 can land, and eps_land at zero drift is a landing floor -- what Rule B compares
K_l*delta_perp against; with raw privates that number is the chart error, ~0.18 at k = 32 on the smoke).  The images
are those of ladder_cell.py (`mlp_letter_a` / `d15_letter_a`) and of the bootstrap; the head is extended by a ZERO row (m+1) and
every private carries the new label, as in those cells.  The chart is the ladder's `pca`: PCA-k of the letter's
TRAIN split (the public pool; disjoint from the test split by construction).
RELEASE.  For every adapted layer l: A0 ~ N(0, 1/n_in) (one Generator(seed+7) over the adapted layers in network
order), B0 = 0, then T full-batch steps of
        v <- mu v + grad,      P <- (1 - eta*lambda) P - eta v          (P in {A, B}; zero initial buffers)
i.e. plain SGD at mu = lambda = 0, momentum with zero buffers (technical record A2), scalar weight decay applied to
the parameter (A <- (1-eta*lambda)A - eta*grad, as the plan states it).  Gradients by autograd; at mu = lambda = 0 the
update equals the ladder's hand-written `B - lr D (A H)^T, A - lr B^T D H^T` with D = (softmax - Y)/N.
MEASURED AT THE TRUTH, per adapted layer (definitions replicated from survival.measure, l = the layer's INPUT):
  delta       = max_{t<T} ||H_{l,t} - H_l^0|| / ||H_l^0||                     (H_l^0 = the base network's input to l)
  delta_perp  = max_{t<T} ||P_{col(H_l^0)^perp}(H_{l,t} - H_l^0)|| / ||H_l^0||   (the T3 axis; in-span drift is free)
  delta_final = the same at t = T (post-final-update; NOT in the training span, recorded, never gated on)
  drift_rank  = rank [H_{l,0}-H_l^0 ... H_{l,T-1}-H_l^0]      N_prime = rank [H_{l,0} ... H_{l,T-1}]   (span_of, 1e-10)
  rank_B_T    = #{sigma(B_T) > 1e-10 sigma_1}                 contaminated = rank_B_T < N_prime  (audit item 2)
  C_full      = P_{row(B_T)^perp} A_T  (rank r - N' expected)     C_trunc = keep only the top-N directions (rank r - N)
  rho_full / rho_trunc = ||C H_l^0|| / (||C|| ||H_l^0||);  rank_C_full = numrank(C_full, ref=||A_T||_F) (absolute floor)
  beta = ||B_T A_T||_2;  diverged = non-finite trajectory OR max rel drift > 1e3 OR beta > 1e6   (survival's gate)
  K_l = rho_trunc / delta_perp  (the Rule-B slope, measured per net -- no prior band, audit item 3)
  seed_scale = (1 - eta*lambda)^T;  rank_C_at_seed_floor = #{sigma(C_full) > 1e-10 * seed_scale * ||A_0||_F}
  rank_C_at_AT_floor = rank_C_full;  the full sigma ladders of B_T and C_full are on the row (rank = ladder + gap)
  res_truth_full / res_truth_trunc = per image ||C phi_l(x_i)|| / ||A_T phi_l(x_i)|| through the BASE map phi_l
THE SOLVE (ladder_cell / bootstrap, same solver, same starts, same criteria).  Objective per certificate in the solve
  f_l(w) = C_l phi_l(psi(w)) / ||A_{T,l} phi_l(psi(w))||   with phi_l = the PUBLIC BASE network's input to layer l
(the attacker has the public model and the release; Prop. A says the full certificate annihilates H_l^0 exactly, which
is the t = 0 point of the training span).  `--stack-below` concatenates the certificate of the adapted layer below the
target (its own input; exact when that input is frozen); `--solve-cert full|trunc|both` picks the certificate(s).
Starts: `--starts` random w0 ~ coord_std * N(0,1) (Generator(seed+31)); LM (certificate.lm_cert, --iters).
References: the raw truth (tier 1: rel. pixel error < 1e-2), and x*_chart = the same LM started at the truth's chart
coordinates (NOT attacker-available; audit item 10a: the chart optimum, never the pixel projection).  Tier 2 =
experiments.utils.perceptual_id.score_image: the truth among 99 public decoys (train split) ranked by SSIM and by the
base model's penultimate features, for the ladder's `found_best` arm (start nearest each truth) and for the attacker's
own candidates (best objective, non-degenerate, de-duplicated at 0.05 as bootstrap.pick_candidates, at most N).
VERDICT per image (the three CLAUDE.md outcomes + the audit's fourth + one this chart forces):
  contaminated            rank_B_T < N_prime at the TARGET layer (before any rank test; no small parameter exists)
  recovered               a start lands on the raw truth (rel err < 1e-2)
  chart-limited           a start reaches x*_chart (rel err < 1e-2) but x*_chart is > 1e-2 from the truth: the solver
                          did its job, the k-chart does not contain the image (the PCA-32 case, bootstrap/ladder)
  alias (residual zero, wrong image)      a non-degenerate start at the truth's FLOOR (objective <= max(1e-20, 100 x the
                          objective at the raw truth)) that is at no image's x*_chart and on no truth -- an information
                          problem.  Bootstrap's weaker `alias_in_chart` (objective <= objective_opt, elsewhere) is a
                          FIELD, not a verdict: on a raw-private PCA chart it fires on residuals ~0.1 (smoke 366163)
  optimisation failure (residual not zero)   nothing reached x*_chart and nothing matched its objective
eps_land (audit item 3) is MEASURED in every cell as max sqrt(objective) over starts that reached some x*_chart; the
zero-drift control is the cell with `--adapt` = `--target` alone (`is_control` on the row); the coordinator joins it
to the drift cells of the same (model, target, T, lr, seed).

PRE-REGISTERED OUTCOMES (written 2026-09-18 BEFORE the smoke ran):
  P3 Rule A (rank).  The full certificate is exact at any drift MAGNITUDE (rho_full at the FP64 floor wherever
    rank_B_T == N_prime) and loses RANK: rank_C_full = min(r, n_l) - N_prime with N_prime = N + drift_rank (orthogonal
    part).  The full-certificate solve can pin the k-chart iff rank_C_full >= k (local); alias-freedom needs k < rank.
    Prediction: reached-x*_chart / landing rate drops to the wrong-release floor exactly where rank_C_full < k,
    independent of delta.  A landing rate that tracks delta (or delta_perp) while rank_C_full >= k refutes Rule A.
  Contaminated branch (audit item 2).  Where rank_B_T < N_prime the residual at the truth is O(1) with no small
    parameter: verdict `contaminated`, keyed on rank_B_T vs N_prime BEFORE any rank test.  Pre-registered cells: the
    strong MLP with the head as target (`--adapt 2 3 --target 3`; rank B_T <= m-1 = 10 < N + drift) is contaminated
    for T > 1; the twin's hidden targets (n_l = 1000) are not expected to be, unless drift_rank >= r - N.
  P3 Rule B (magnitude), truncated certificate.  rho_trunc = K_l * delta_perp with K_l measured per net (one synthetic
    net gave 0.082; T3's closed form gives ~0.5 in slow drift; NO band is pre-registered).  For `--solve-cert trunc`
    the landing rate against K_l*delta_perp/eps_land is predicted to be a single step curve across (T, lr, r_lower);
    a curve that is monotone in delta but not in delta_perp refutes the T3 axis.  Deliverables: K_l per net, its
    stability across cells, and whether the step curve is single-valued.
  P6 momentum (zero buffers).  Exact prediction: rho_full at the floor, rank_C_full / gap unchanged within seed
    spread.  Under drift (P3 cell) the CONJECTURE (marked so) that momentum raises N_prime faster per step acts only
    through the upstream adapter (N' is the span of the layer INPUTS).
  P6 weight decay.  Exact prediction: the off-span block is (1 - eta*lambda)^T A_0, so `rho_full` stays at the floor
    and `rank_C_at_seed_floor` = r - N_prime at EVERY finite T; only the eta*lambda = 1 corner collapses C (rank 0,
    Counterexample 5.1).  Numerical (deployment) prediction: `rank_C_at_AT_floor` falls toward 0 as seed_scale*||A_0||
    drops below 1e-10 ||A_T||, i.e. at eta*lambda*T ~ 23; the row reports both ranks separately.  Expected ORDER of
    events as eta*lambda*T grows: residual at floor + rank unchanged (exact) -> rank loss at the ||A_T|| floor only ->
    solve failure when rank_C_at_AT_floor < k -> C = 0 at the corner.

  python -u -m experiments.multilayer_cert.drift_cert --adapt 1 2 --target 2 --r-per-layer 16 64 --T 1 20 \\
         --lr 0.01 --starts 20 --seed 1 --out results/multilayer_cert/drift_cert_smoke.jsonl
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch
import torch.nn.functional as F

from experiments.exact_inversion.deep_stack import inputs_of, load_deep, GELU
from experiments.exact_inversion.new_class import load_emnist_letters
from experiments.exact_inversion.trained_backbone import read_idx
from experiments.exact_inversion.certificate import lm_cert
from experiments.multilayer_cert.common import certificate, numrank, rel_annihilation, span_of, provenance

torch.set_default_dtype(torch.float64)

LAND = 1e-2            # tier-1 exact landing (ladder / bootstrap / RECOVER_TOL)
FLOOR = 1e-20          # certificate.py's at-floor threshold on the objective ||C phi||^2/||A_T phi||^2
DEGEN = 0.05           # feature-norm ratio below which a start is a blank image (certificate.py / ladder)
DEDUPE = 0.05          # relative distance below which two chart images are the same candidate (bootstrap.pick_candidates)
DRIFT_MAX, BETA_MAX = 1e3, 1e6     # survival.py's divergence gate
TOL = 1e-10            # survival.py's certificate / span tolerance


def log(s): print(s, flush=True)


# ------------------------------------------------------------------------------------------------ model + data
def load_model(path, dev):
    """(Ws, b1, meta) in the deep_stack convention for either checkpoint layout."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "Ws" in blob:                                                       # deep_stack checkpoint (ordered list)
        Ws, b1, ck = load_deep(path, dev)
        meta = {k: v for k, v in ck.items() if k not in ("Ws", "b1") and not torch.is_tensor(v)}
        meta["layout"] = "deep_stack"
    else:                                                                  # 784-1000-1000-10 state_dict (TrainedBackbone layout)
        sd = blob["state_dict"] if "state_dict" in blob else blob
        keys = sorted([k for k in sd if k.endswith(".weight")], key=lambda s: int(s.split(".")[1]))
        Ws = [sd[k].to(dev).double() for k in keys]; b1 = sd["layers.0.bias"].to(dev).double()
        assert all(not k.endswith(".bias") or k == "layers.0.bias" for k in sd), "bias beyond layer 1: not the deep_stack family"
        meta = dict(layout="state_dict", test_acc=blob.get("test_acc"), epoch=blob.get("epoch"))
    return Ws, b1, meta


def feats_upto(x, Ws, b1, lmax):
    """[h^0 = x, h^1, ..., h^{lmax}] of the BASE network (adapters off): inputs to layers 0..lmax (0-indexed)."""
    hs = [x]; h = x
    for l in range(lmax):
        h = GELU(Ws[l] @ h + (b1[:, None] if l == 0 else 0)); hs.append(h)
    return hs


def train_lora(X, y, Ws, b1, adapt, A0s, T, lr, mom, wd):
    """Adapter-masked LoRA training (adapt = 0-indexed layers).  Returns As, Bs (None where frozen), reps[t][j] =
    input to adapted layer adapt[j] at step t < T (pre-update), reps_final[j] = the same after the last update."""
    D = len(Ws); N = X.shape[1]
    As, Bs, vA, vB = [None] * D, [None] * D, {}, {}
    for j, i in enumerate(adapt):
        As[i] = A0s[j].clone().requires_grad_(True)
        Bs[i] = torch.zeros(Ws[i].shape[0], A0s[j].shape[0], dtype=X.dtype, device=X.device, requires_grad=True)
        vA[i] = torch.zeros_like(As[i]); vB[i] = torch.zeros_like(Bs[i])

    def fwd(As_, Bs_):
        hs = inputs_of(X, Ws, b1, As_, Bs_); h = hs[-1]; z = Ws[-1] @ h
        if As_[-1] is not None: z = z + Bs_[-1] @ (As_[-1] @ h)
        return hs, z
    reps = []
    with torch.enable_grad():
        for t in range(T):
            hs, z = fwd(As, Bs)
            reps.append([hs[i].detach().clone() for i in adapt])
            loss = F.cross_entropy(z.T, y)                                  # mean CE: grad = (softmax - Y)/N exactly
            params = [As[i] for i in adapt] + [Bs[i] for i in adapt]
            gs = torch.autograd.grad(loss, params); n = len(adapt)
            for j, i in enumerate(adapt):
                vA[i] = mom * vA[i] + gs[j]; vB[i] = mom * vB[i] + gs[n + j]
                As[i] = ((1.0 - lr * wd) * As[i].detach() - lr * vA[i]).requires_grad_(True)
                Bs[i] = ((1.0 - lr * wd) * Bs[i].detach() - lr * vB[i]).requires_grad_(True)
    As = [None if a is None else a.detach() for a in As]; Bs = [None if b is None else b.detach() for b in Bs]
    with torch.no_grad():
        hs, _ = fwd(As, Bs); reps_final = [hs[i] for i in adapt]
    return As, Bs, reps, reps_final


def ladder(s, around, width=3):
    """A few singular values around a cut, for the row (rank = ladder + gap, never an integer alone)."""
    s = [float(v) for v in s]; lo, hi = max(0, around - width), min(len(s), around + width)
    return dict(idx_from=lo, values=s[lo:hi])


def layer_stats(layer1, j, H0, reps, reps_final, A, B, A0, N, r, lr, wd, T, X_raw_feats):
    """survival.measure's per-layer block, plus the P6 two-floor ranks and the per-image residuals at the truth."""
    P0 = torch.linalg.qr(H0)[0][:, :H0.shape[1]]
    Dts = [rt[j] - H0 for rt in reps]
    delta = max(float(d.norm() / H0.norm()) for d in Dts)
    dperp = max(float((d - P0 @ (P0.T @ d)).norm() / H0.norm()) for d in Dts)
    delta_final = float((reps_final[j] - H0).norm() / H0.norm())
    drank = span_of(Dts)[0] if delta > 0 else 0
    Hcat = torch.cat([rt[j] for rt in reps], 1); sH = torch.linalg.svdvals(Hcat)
    Nprime = int((sH > TOL * sH[0]).sum())                                  # == span_of(...)[0], with its ladder kept
    Cf, q, sB = certificate(A, B, tol=TOL); Ct, _, _ = certificate(A, B, keep=N, tol=TOL)
    gap = float(sB[N - 1] / sB[N]) if len(sB) > N and float(sB[N]) > 0 else float("inf")
    rank_B_gap = float(sB[q - 1] / sB[q]) if q < len(sB) and float(sB[q]) > 0 else float("inf")
    rkC, sC = numrank(Cf, rtol=TOL, ref=float(A.norm()))
    seed_scale = (1.0 - lr * wd) ** T
    seed_floor = TOL * seed_scale * float(A0.norm())
    rk_seed = int((sC > seed_floor).sum())
    beta = float(torch.linalg.matrix_norm(B @ A, 2))
    with torch.no_grad():
        Hr = X_raw_feats
        rf = torch.linalg.norm(Cf @ Hr, dim=0) / torch.linalg.norm(A @ Hr, dim=0)
        rt_ = torch.linalg.norm(Ct @ Hr, dim=0) / torch.linalg.norm(A @ Hr, dim=0)
    return dict(layer=layer1, n_in=int(H0.shape[0]), n_out=int(B.shape[0]), r=r, delta=delta, delta_perp=dperp, delta_final=delta_final,
                drift_rank=drank, N_prime=Nprime, N_prime_gap=(float(sH[Nprime - 1] / sH[Nprime]) if Nprime < len(sH) and float(sH[Nprime]) > 0 else float("inf")),
                N_prime_ladder=ladder(sH, Nprime), rank_B_T=q, rank_B_gap=rank_B_gap, gap_N=gap, B_T_sigma=[float(v) for v in sB],
                B3_holds=bool(q == Nprime), contaminated=bool(q < Nprime), beta=beta,
                rank_C_full=rkC, rank_C_at_AT_floor=rkC, rank_C_at_seed_floor=rk_seed, C_sigma=[float(v) for v in sC],
                expect_rank_C=max(0, min(r - Nprime, int(H0.shape[0]) - Nprime)),
                seed_scale=seed_scale, seed_floor_abs=seed_floor, seed_floor_below_noise=bool(seed_floor < 1e-15 * float(A.norm())),
                A0_fro=float(A0.norm()), A_T_fro=float(A.norm()), B_T_fro=float(B.norm()),
                rho_full=rel_annihilation(Cf, H0), rho_trunc=rel_annihilation(Ct, H0),
                K_l=(rel_annihilation(Ct, H0) / dperp if dperp > 0 else None),
                res_truth_full=[float(v) for v in rf], res_truth_trunc=[float(v) for v in rt_]), Cf, Ct


def rel_err_cols(x, X): return torch.linalg.norm(X - x[:, None], dim=0) / torch.linalg.norm(X, dim=0)


def grid_png(path, rows, N, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(rows), N, figsize=(1.1 * N + 0.4, 1.2 * len(rows) + 0.6), squeeze=False)
    for ri, (lab, X) in enumerate(rows):
        for c in range(N):
            a_ = ax[ri, c]; a_.axis("off")
            if X is not None and c < X.shape[1]:
                a_.imshow(X[:, c].reshape(28, 28).clamp(0, 1).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            if c == 0: a_.set_title(lab, fontsize=6, loc="left")
    fig.suptitle(title, fontsize=6); fig.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110); plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_d15w1000_full.pth")
    ap.add_argument("--adapt", nargs="+", type=int, required=True, help="1-indexed layers carrying a LoRA adapter (e.g. 1 2)")
    ap.add_argument("--target", type=int, required=True, help="1-indexed layer whose certificate is solved (must be in --adapt)")
    ap.add_argument("--r-per-layer", nargs="+", type=int, required=True, help="rank per adapted layer, same order as --adapt")
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--T", nargs="+", type=int, default=[400]); ap.add_argument("--lr", nargs="+", type=float, default=[0.01])
    ap.add_argument("--momentum", nargs="+", type=float, default=[0.0]); ap.add_argument("--wd", nargs="+", type=float, default=[0.0])
    ap.add_argument("--seed", nargs="+", type=int, default=[1])
    ap.add_argument("--chart", choices=["pca"], default="pca", help="PCA-k of the public pool of the added class (the ladder's attacker-available chart)")
    ap.add_argument("--images", choices=["letter_a"], default="letter_a", help="the 18 Sept package's eight: EMNIST letter a, test split, randperm(seed+7)[:N]")
    ap.add_argument("--private", choices=["raw", "onchart"], default="raw", help="raw: the release is trained on the pixels (the truth is NOT in the "
                    "PCA-k chart; x*_chart is the reference). onchart: trained on the chart projections psi(coords(x)), so the truth IS in the chart, "
                    "tier 1 can land and eps_land at zero drift is a landing floor rather than the chart error (Rule B needs this)")
    ap.add_argument("--starts", type=int, default=400); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--solve", dest="solve", action="store_true", default=True); ap.add_argument("--no-solve", dest="solve", action="store_false")
    ap.add_argument("--stack-below", action="store_true", help="add the certificate of the adapted layer just below the target to the objective")
    ap.add_argument("--solve-cert", choices=["full", "trunc", "both"], default="full")
    ap.add_argument("--label", default="")
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True); ap.add_argument("--save-dir", default=None); ap.add_argument("--fig-dir", default="figures/multilayer_cert/drift_cert")
    a = ap.parse_args(); dev = torch.device(a.device)
    assert len(a.adapt) == len(a.r_per_layer), "--r-per-layer must match --adapt"
    assert a.target in a.adapt, "--target must be one of --adapt"
    assert sorted(a.adapt) == a.adapt and len(set(a.adapt)) == len(a.adapt), "--adapt must be strictly increasing"
    PROV = provenance(__file__); job = os.environ.get("LSB_JOBID", "manual")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_dir = a.save_dir or os.path.join(os.path.dirname(a.out) or ".", "drift_cert"); os.makedirs(save_dir, exist_ok=True)

    Ws, b1, meta = load_model(a.model, dev); D = len(Ws)
    assert 1 <= a.target <= D and all(1 <= l <= D for l in a.adapt), f"layers must be in 1..{D}"
    # ---- WP0-style gate: the loaded net must classify MNIST (catches a mis-assembled loader before any number is produced)
    Xte, yte = read_idx(a.data_root, "test")
    with torch.no_grad():
        Zt = Ws[-1] @ feats_upto(torch.tensor(Xte[:2000], device=dev).T, Ws, b1, D - 1)[-1]
        gate_acc = float((Zt.argmax(0).cpu() == torch.tensor(yte[:2000])).double().mean())
    ck_meta = {k_: meta.get(k_) for k_ in ("test_acc", "train_acc", "train_loss")}
    log(f"# drift_cert  model={a.model} layout={meta.get('layout')} depth={D} ckpt_meta={ck_meta}"
        f"  gate test acc (2000) = {gate_acc*100:.2f}%  {'PASS' if gate_acc > 0.9 else 'FAIL -- loader wrong'}  git={PROV['git']} sha={PROV['script_sha']} job={job} host={socket.gethostname()}")
    if gate_acc <= 0.9: sys.exit(3)
    # ---- privates + chart (ladder_cell verbatim: letter a, extend the head by a zero row, PCA of the train split)
    fl = load_emnist_letters(a.data_root, "a")
    Pub = torch.tensor(fl["train"][0], device=dev); Pri = torch.tensor(fl["test"][0], device=dev)
    n_head = Ws[-1].shape[1]; Ws = Ws[:-1] + [torch.cat([Ws[-1], torch.zeros(1, n_head, device=dev)], 0)]; m = Ws[-1].shape[0]
    mean = Pub.mean(0); _, S_, Vh_ = torch.linalg.svd(Pub - mean, full_matrices=False); V = Vh_[: a.k].T.contiguous()
    psi = lambda Z: mean[:, None] + V @ Z; coords = lambda X: V.T @ (X - mean[:, None])
    coord_std = coords(Pub[:5000].T).std(dim=1, keepdim=True)
    chart_explained = float((S_[: a.k] ** 2).sum() / (S_ ** 2).sum())
    adapt0 = [l - 1 for l in a.adapt]; tgt0 = a.target - 1; jt = adapt0.index(tgt0)
    below = [l for l in a.adapt if l < a.target]; below1 = max(below) if below else None
    is_control = (a.adapt == [a.target])
    solve_layers = [a.target] + ([below1] if (a.stack_below and below1 is not None) else [])
    if a.stack_below and below1 is None: log("# --stack-below requested but no adapted layer lies below the target: single certificate")
    cells = [(seed, T, lr, mom, wd) for seed in a.seed for T in a.T for lr in a.lr for mom in a.momentum for wd in a.wd]
    log(f"# adapt={a.adapt} r={a.r_per_layer} target={a.target} control={is_control} solve_layers={solve_layers} cert={a.solve_cert} "
        f"k={a.k} chart=pca (explained {chart_explained:.3f}) starts={a.starts} cells={len(cells)} m={m} (head extended)")
    if os.path.basename(a.model).startswith("mnist_mlp_strong") and a.target == D:
        log("# PRE-REGISTERED contaminated cell: the target is the softmax head (rank B_T <= m-1 < N + drift for T > 1)")

    def emit(row):
        with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    from experiments.utils.perceptual_id import score_image, feature_extractor     # CPU line-up: truth + 99 public decoys
    feat_cpu = feature_extractor(a.model); key = ("emnist", "a"); shape = (1, 28, 28)

    for (seed, T, lr, mom, wd) in cells:
        t_cell = time.time()
        g = torch.Generator().manual_seed(seed + 7); perm = torch.randperm(Pri.shape[0], generator=g)
        X_pix = Pri[perm[: a.N]].T.contiguous(); join_idx = [int(v) for v in perm[: a.N]]
        X_raw = psi(coords(X_pix)) if a.private == "onchart" else X_pix           # the TRUTH the release is trained on and scored against
        y = torch.full((a.N,), m - 1, device=dev, dtype=torch.long)
        gA = torch.Generator().manual_seed(seed + 7)
        A0s = [(torch.randn(r, Ws[i].shape[1], generator=gA) / math.sqrt(Ws[i].shape[1])).to(dev) for i, r in zip(adapt0, a.r_per_layer)]
        H0s = feats_upto(X_raw, Ws, b1, max(adapt0))                        # base inputs to every layer up to the deepest adapted one
        As, Bs, reps, reps_final = train_lora(X_raw, y, Ws, b1, adapt0, A0s, T, lr, mom, wd)
        cfg = dict(part="drift_cert", label=a.label, model=a.model, model_layout=meta.get("layout"), depth=D, adapt=a.adapt, target=a.target,
                   r_per_layer=a.r_per_layer, r_target=a.r_per_layer[jt], layer_below=below1, is_control=is_control, N=a.N, k=a.k, chart="pca",
                   chart_explained=chart_explained, T=T, lr=lr, momentum=mom, wd=wd, eta_lambda=lr * wd, eta_lambda_T=lr * wd * T,
                   seed=seed, images=a.images, private=a.private, private_join_idx=join_idx,
                   private_chart_err=[float(torch.linalg.norm(psi(coords(X_pix))[:, i] - X_pix[:, i]) / torch.linalg.norm(X_pix[:, i])) for i in range(a.N)], m=m, head_extended=True, label_new=m - 1, starts=a.starts, iters=a.iters,
                   solve=a.solve, stack_below=bool(a.stack_below and below1 is not None), solve_layers=solve_layers, solve_cert=a.solve_cert,
                   gate_test_acc=gate_acc, ckpt_meta={k_: meta.get(k_) for k_ in ("test_acc", "train_acc", "train_loss")},
                   git=PROV["git"], script_sha=PROV["script_sha"], job=job, host=socket.gethostname(), cmd=" ".join(sys.argv))
        # ---- divergence gate (survival.measure, verbatim thresholds)
        finite = all(torch.isfinite(h).all() for rt in reps + [reps_final] for h in rt) and all(torch.isfinite(t).all() for t in [As[i] for i in adapt0] + [Bs[i] for i in adapt0])
        rel = [max(float((rt[j] - H0s[i]).norm() / H0s[i].norm()) for rt in reps + [reps_final]) if finite else float("inf") for j, i in enumerate(adapt0)]
        betas = []
        for i in adapt0:
            try:
                M = Bs[i] @ As[i]; betas.append(float(torch.linalg.matrix_norm(M, 2)) if torch.isfinite(M).all() else float("inf"))
            except Exception: betas.append(float("inf"))
        diverged = (not finite) or (not all(math.isfinite(v) for v in rel)) or max(rel) > DRIFT_MAX or max(betas) > BETA_MAX
        if diverged:
            row = dict(cfg, diverged=True, layers=[dict(layer=l, rel_drift_max=(v if math.isfinite(v) else None), beta=(b if math.isfinite(b) else None))
                                                   for l, v, b in zip(a.adapt, rel, betas)], seconds=time.time() - t_cell)
            emit(row); log(f"  seed={seed} T={T} lr={lr} mom={mom} wd={wd}: DIVERGED (rel drift {rel}, beta {betas}) -- recorded, skipped"); continue
        # ---- per-layer statistics at the truth
        layers, Cs = [], {}
        for j, i in enumerate(adapt0):
            st, Cf, Ct = layer_stats(i + 1, j, H0s[i], reps, reps_final, As[i], Bs[i], A0s[j], a.N, a.r_per_layer[j], lr, wd, T, H0s[i])
            layers.append(st); Cs[i + 1] = dict(full=Cf, trunc=Ct, A=As[i])
        tg = layers[jt]
        row = dict(cfg, diverged=False, layers=layers, target_stats={k_: tg[k_] for k_ in ("delta", "delta_perp", "drift_rank", "N_prime", "rank_B_T", "rank_C_full",
                                                                                            "rank_C_at_seed_floor", "rho_full", "rho_trunc", "contaminated", "K_l", "seed_scale", "beta")},
                   contaminated=tg["contaminated"], rank_ok_for_k=bool(tg["rank_C_full"] >= a.k), alias_free_k=bool(a.k < tg["rank_C_full"]))
        log(f"  seed={seed} T={T} lr={lr} mom={mom} wd={wd}: " + " | ".join(
            f"l{s['layer']} d={s['delta']:.2e}/perp {s['delta_perp']:.2e} drk={s['drift_rank']} N'={s['N_prime']} rkB={s['rank_B_T']} rkC={s['rank_C_full']}"
            f"(seedfloor {s['rank_C_at_seed_floor']}) full={s['rho_full']:.1e} trunc={s['rho_trunc']:.1e}{' CONTAMINATED' if s['contaminated'] else ''}" for s in layers)
            + f"  [{time.time()-t_cell:.1f}s]")
        if not a.solve:
            row["seconds"] = time.time() - t_cell; emit(row); continue
        # ---- the solve
        kinds = ["full", "trunc"] if a.solve_cert == "both" else [a.solve_cert]
        solves = {}
        for kind in kinds:
            t_s = time.time()
            certs = [(l - 1, Cs[l][kind], Cs[l]["A"]) for l in solve_layers]; lmax = max(i for i, _, _ in certs)

            def fun(w, certs=certs, lmax=lmax):
                hs = feats_upto(psi(w.reshape(a.k, 1)), Ws, b1, lmax)
                return torch.cat([(C @ hs[i]).reshape(-1) / torch.linalg.norm(A @ hs[i]) for i, C, A in certs])
            A_t = Cs[a.target]["A"]
            with torch.no_grad():
                feat_ref = float(torch.linalg.norm(A_t @ feats_upto(Pub[:256].T, Ws, b1, tgt0)[-1], dim=0).median())
                Wtrue = coords(X_raw); X_proj = psi(Wtrue)
                obj_proj = [float(fun(Wtrue[:, i]) @ fun(Wtrue[:, i])) for i in range(a.N)]
                obj_raw = [float(v) for v in (torch.linalg.norm(Cs[a.target][kind] @ H0s[tgt0], dim=0) / torch.linalg.norm(A_t @ H0s[tgt0], dim=0)) ** 2]
                proj_err = [float(torch.linalg.norm(X_proj[:, i] - X_raw[:, i]) / torch.linalg.norm(X_raw[:, i])) for i in range(a.N)]   # the chart's own ceiling
            # x*_chart per truth: the same LM from the ORACLE start (coords of the truth); NOT attacker-available
            opt = []
            for i in range(a.N):
                w, obj, it = lm_cert(fun, Wtrue[:, i].clone(), a.iters)
                with torch.no_grad(): x = psi(w.reshape(a.k, 1))[:, 0]
                opt.append(dict(w=w.detach(), x=x, objective=obj, iters=it, err_truth=float(rel_err_cols(x, X_raw)[i])))
            X_opt = torch.stack([o["x"] for o in opt], 1)
            gs = torch.Generator().manual_seed(seed + 31); runs = []; Wfound = []
            for s in range(a.starts):
                w0 = (torch.randn(a.k, 1, generator=gs).to(dev) * coord_std).reshape(-1)
                w, obj, it = lm_cert(fun, w0, a.iters)
                with torch.no_grad():
                    x = psi(w.reshape(a.k, 1))[:, 0]
                    e_raw = rel_err_cols(x, X_raw); e_opt = rel_err_cols(x, X_opt)
                    fr = float(torch.linalg.norm(A_t @ feats_upto(x.reshape(-1, 1), Ws, b1, tgt0)[-1]) / feat_ref)
                jr, jo = int(e_raw.argmin()), int(e_opt.argmin())
                runs.append(dict(objective=obj, iters=it, nearest=jr, err=float(e_raw[jr]), landed=bool(e_raw[jr] < LAND),
                                 nearest_opt=jo, err_opt=float(e_opt[jo]), reached_opt=bool(e_opt[jo] < LAND), feat_ratio=fr, degenerate=bool(fr < DEGEN)))
                Wfound.append(w.detach())
                if (s + 1) % 100 == 0 or s + 1 == a.starts:
                    log(f"     [{kind}] {s+1}/{a.starts} starts {time.time()-t_s:.0f}s  landed {sum(r['landed'] for r in runs)}  reached x*chart {sum(r['reached_opt'] for r in runs)}")
            X_found = psi(torch.stack(Wfound, 1))
            with torch.no_grad():
                E_raw = torch.stack([rel_err_cols(X_raw[:, i], X_found) for i in range(a.N)], 0)      # (N, starts)
                E_opt = torch.stack([rel_err_cols(X_opt[:, i], X_found) for i in range(a.N)], 0)
            best_raw = E_raw.argmin(1); best_opt = E_opt.argmin(1)
            # attacker's own candidates: best objective, non-degenerate, de-duplicated, at most N (bootstrap.pick_candidates)
            order = sorted([s for s in range(a.starts) if not runs[s]["degenerate"]], key=lambda s: runs[s]["objective"]) or sorted(range(a.starts), key=lambda s: runs[s]["objective"])
            cand = []
            for s in order:
                if all(float(rel_err_cols(X_found[:, s], X_found[:, c:c + 1])[0]) > DEDUPE for c in cand): cand.append(s)
                if len(cand) == a.N: break
            X_cand = X_found[:, cand]
            att_for = [int(rel_err_cols(X_raw[:, i], X_cand).argmin()) for i in range(a.N)]     # nearest candidate to each truth (coverage)
            X_att = X_cand[:, att_for]
            per = []
            for i in range(a.N):
                bi, oi = int(best_raw[i]), int(best_opt[i])
                e_r, e_o = float(E_raw[i, bi]), float(E_opt[i, oi])
                landed_i, reached_i = bool(e_r < LAND), bool(e_o < LAND)
                at_opt_obj = [s for s in range(a.starts) if not runs[s]["degenerate"] and runs[s]["objective"] <= opt[i]["objective"] * (1 + 1e-6) + FLOOR]
                below_opt = [s for s in at_opt_obj if not runs[s]["reached_opt"] and not runs[s]["landed"]]      # bootstrap's alias_in_chart
                floor_i = max(FLOOR, 1e2 * obj_raw[i])                                                          # "residual zero" = at the truth's floor
                alias_starts = [s for s in below_opt if runs[s]["objective"] <= floor_i]
                if tg["contaminated"]: verdict = "contaminated"
                elif landed_i: verdict = "recovered"
                elif reached_i: verdict = "recovered" if opt[i]["err_truth"] < LAND else "chart-limited"
                elif alias_starts: verdict = "alias (residual zero, wrong image)"
                else: verdict = "optimisation failure (residual not zero)"
                t2_found = score_image(X_found[:, bi].cpu(), X_raw[:, i].cpu(), key, shape, feat_cpu)
                t2_att = score_image(X_att[:, i].cpu(), X_raw[:, i].cpu(), key, shape, feat_cpu)
                per.append(dict(i=i, verdict=verdict, best_err=e_r, landed=landed_i, best_start=bi, best_objective=runs[bi]["objective"],
                                err_opt=e_o, reached_opt=reached_i, best_start_opt=oi, objective_opt=opt[i]["objective"], opt_iters=opt[i]["iters"],
                                err_opt_truth=opt[i]["err_truth"], objective_at_projection=obj_proj[i], objective_at_raw_truth=obj_raw[i],
                                n_starts_at_opt_objective=len(at_opt_obj), n_starts_below_opt_elsewhere=len(below_opt), alias_in_chart=bool(below_opt and not reached_i),
                                n_alias_starts=len(alias_starts), floor_objective=floor_i,
                                res_truth=math.sqrt(obj_raw[i]), res_opt=math.sqrt(max(opt[i]["objective"], 0.0)),
                                attacker_candidate=cand[att_for[i]], attacker_err=float(rel_err_cols(X_raw[:, i], X_att[:, i:i + 1])[0]),
                                tier2_found_best={k_: t2_found[k_] for k_ in ("ssim_truth", "ssim_control", "rank_ssim", "rank_l2", "rank_feat_l2", "rank_feat_cos", "top1_ssim", "top5_ssim", "top1_feat", "top5_feat")},
                                tier2_attacker={k_: t2_att[k_] for k_ in ("ssim_truth", "ssim_control", "rank_ssim", "rank_l2", "rank_feat_l2", "rank_feat_cos", "top1_ssim", "top5_ssim", "top1_feat", "top5_feat")}))
            reached_any = [s for s in range(a.starts) if runs[s]["reached_opt"] and not runs[s]["degenerate"]]
            eps_land = max(math.sqrt(max(runs[s]["objective"], 0.0)) for s in reached_any) if reached_any else None
            cnt = lambda arm, k_: int(sum(bool(p[arm][k_]) for p in per))
            solves[kind] = dict(kind=kind, landed=int(sum(r["landed"] for r in runs)), images_found=int(sum(p["landed"] for p in per)),
                                reached_opt=int(sum(r["reached_opt"] for r in runs)), images_reached_opt=int(sum(p["reached_opt"] for p in per)),
                                n_degenerate=int(sum(r["degenerate"] for r in runs)), objective_min=float(min(runs[s]["objective"] for s in (order or range(a.starts)))),
                                objective_median=float(np.median([r["objective"] for r in runs])),
                                eps_land=eps_land, res_opt_max=max(p["res_opt"] for p in per), res_truth_max=max(p["res_truth"] for p in per),
                                err_opt_truth_median=float(np.median([p["err_opt_truth"] for p in per])), proj_err_median=float(np.median(proj_err)), proj_err_per_image=proj_err,
                                verdicts={v: int(sum(p["verdict"] == v for p in per)) for v in ("recovered", "chart-limited", "alias (residual zero, wrong image)", "optimisation failure (residual not zero)", "contaminated")},
                                tier2_found_best=dict(top1_ssim=cnt("tier2_found_best", "top1_ssim"), top5_ssim=cnt("tier2_found_best", "top5_ssim"), top1_feat=cnt("tier2_found_best", "top1_feat"), top5_feat=cnt("tier2_found_best", "top5_feat"),
                                                      ssim_truth_median=float(np.median([p["tier2_found_best"]["ssim_truth"] for p in per])), ssim_control_median=float(np.median([p["tier2_found_best"]["ssim_control"] for p in per]))),
                                tier2_attacker=dict(top1_ssim=cnt("tier2_attacker", "top1_ssim"), top5_ssim=cnt("tier2_attacker", "top5_ssim"), top1_feat=cnt("tier2_attacker", "top1_feat"), top5_feat=cnt("tier2_attacker", "top5_feat"),
                                                    ssim_truth_median=float(np.median([p["tier2_attacker"]["ssim_truth"] for p in per])), n_candidates=len(cand), truths_covered=len(set(att_for))),
                                per_image=per, runs=runs, seconds=time.time() - t_s)
            tag = f"{os.path.splitext(os.path.basename(a.model))[0]}_adapt{'-'.join(map(str, a.adapt))}_tgt{a.target}_r{'-'.join(map(str, a.r_per_layer))}_T{T}_lr{lr:g}_m{mom:g}_wd{wd:g}_s{seed}{'_stack' if row['stack_below'] else ''}_{kind}_{job}"
            torch.save(dict(x_raw=X_raw.cpu(), x_pix=X_pix.cpu(), private=a.private, x_proj=X_proj.cpu(), x_opt=X_opt.cpu(), x_found_best=X_found[:, best_raw].cpu(), x_found_best_opt=X_found[:, best_opt].cpu(),
                            x_attacker=X_att.cpu(), x_cand=X_cand.cpu(), W=torch.stack(Wfound, 1).cpu(), runs=runs, chart_mean=mean.cpu(), chart_V=V.cpu(),
                            A_T={l: As[l - 1].cpu() for l in a.adapt}, B_T={l: Bs[l - 1].cpu() for l in a.adapt}, A0={l: A0s[j].cpu() for j, l in enumerate(a.adapt)},
                            C={l: Cs[l][kind].cpu() for l in solve_layers}, row=dict(row, solve_summary={k_: v for k_, v in solves[kind].items() if k_ not in ("runs",)})),
                       os.path.join(save_dir, tag + ".pth"))
            grid_png(os.path.join(a.fig_dir, tag + ".png"), [("truth", X_raw), ("x*_chart (oracle start)", X_opt), ("found (nearest start)", X_found[:, best_raw]), ("attacker candidate", X_att)],
                     a.N, f"{tag}\nlanded {solves[kind]['landed']}/{a.starts} reached x*chart {solves[kind]['reached_opt']} verdicts {solves[kind]['verdicts']}")
            sv = solves[kind]; eps_str = "None" if sv["eps_land"] is None else f"{sv['eps_land']:.2e}"
            log(f"     [{kind}] landed {sv['landed']}/{a.starts} (images {sv['images_found']}/{a.N}) reached x*chart {sv['reached_opt']} (images {sv['images_reached_opt']}/{a.N}) "
                f"eps_land {eps_str} res_opt max {sv['res_opt_max']:.2e} res_truth max {sv['res_truth_max']:.2e} "
                f"tier2 found top1 ssim/feat {sv['tier2_found_best']['top1_ssim']}/{sv['tier2_found_best']['top1_feat']} attacker {sv['tier2_attacker']['top1_ssim']}/{sv['tier2_attacker']['top1_feat']} "
                f"verdicts {sv['verdicts']} [{sv['seconds']:.0f}s]")
        row["solves"] = {k_: {kk: vv for kk, vv in v.items() if kk != "runs"} for k_, v in solves.items()}
        row["solve_runs"] = {k_: v["runs"] for k_, v in solves.items()}
        row["seconds"] = time.time() - t_cell
        emit(row)
    log(f"# DONE {len(cells)} cells -> {a.out}")


if __name__ == "__main__":
    main()
