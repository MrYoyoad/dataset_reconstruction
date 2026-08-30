"""Ecosystem attack — LOO common-mode subtraction + NN-retrieval GAIN (plan §3/§5, all six gates).

Consumes results/eco_zoo/eco_bank.pth (weak-signal multi-task disjoint zoo). For each TARGET adapter:
  • shared subspace = top-p PCs of the OTHER-TASK adapters' flattened ΔW (leave-one-out, disjoint content);
  • residual ΔW = target − its projection onto that subspace;
  • NN-RETRIEVAL of the target's PRIVATE images from a pool (private + distractors), score = ‖ΔW·(x−μ)‖,
    scored by AUC; GAIN = AUC(residual) − AUC(raw adapter); random-adapter AUC = the gate-4 baseline.
GATES reported (not asserted): (1) PROJECTION test — target-ΔW energy fraction in the shared subspace (and
top-singular-dir projection); (3) DISJOINT-content overlap = target digits ∩ population digits (0 by design);
(2) HEADROOM = raw-adapter AUC mid-range; (4) random-adapter baseline; GAIN CI = cluster-robust t_{G−1} over
the 5 tasks. Observe-framed, weakest→population-attacker. numpy/scipy only. bsub or light-CPU.
"""
import argparse, os, numpy as np, torch
from scipy import stats

BANK = "results/eco_zoo/eco_bank.pth"
P_SHARED = 6          # shared-subspace dimension (θ0 common-mode across other tasks)
RNG = np.random.default_rng(0)


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    bank = [c for c in d["bank"] if c.get("converged", True)]
    for c in bank:
        A = c["A"].to(torch.float64).numpy(); B = c["B"].to(torch.float64).numpy()
        c["dW"] = B @ A                                  # out×in
        c["dw_flat"] = c["dW"].ravel()
        c["priv"] = c["priv_imgs"].to(torch.float64).numpy()   # (N,in) raw private images
    return bank, d["ds_mean"].to(torch.float64).numpy(), d["meta"]


def _auc(labels, scores):
    """Mann-Whitney AUC (P[score(pos) > score(neg)])."""
    order = np.argsort(scores); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1; npos = pos.sum(); nneg = (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def retrieval_auc(dW, pool, labels, mu):
    scores = np.linalg.norm((pool - mu) @ dW.T, axis=1)   # ‖ΔW·(x−μ)‖ per pool image
    return _auc(labels, scores)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bank", default=BANK); ap.add_argument("--p", type=int, default=P_SHARED)
    args = ap.parse_args()
    bank, mu, meta = load(args.bank)
    tasks = [tuple(c["task"]) for c in bank]
    print(f"[eco] {len(bank)} converged adapters | tasks={sorted(set(tasks))} p_shared={args.p} N={meta['N']}")

    per_target = []   # (task, auc_raw, auc_res, gain, auc_rand, proj_frac, proj_top, overlap)
    for ti, tgt in enumerate(bank):
        t_task = tuple(tgt["task"])
        others = [c for c in bank if tuple(c["task"]) != t_task]           # LEAVE-ONE-OUT by TASK (disjoint)
        X = np.stack([c["dw_flat"] for c in others]); Xm = X.mean(0)
        _, _, Vt = np.linalg.svd(X - Xm, full_matrices=False)
        basis = Vt[:args.p]                                                # shared subspace (p × D)
        v = tgt["dw_flat"]
        coeff = basis @ v
        proj_frac = float((coeff @ coeff) / (v @ v + 1e-30))              # GATE 1: energy in shared subspace
        # top private singular direction projection
        U1, s1, V1t = np.linalg.svd(tgt["dW"], full_matrices=False)
        top = np.outer(U1[:, 0], V1t[0]).ravel()
        proj_top = float(((basis @ top) ** 2).sum() / (top @ top + 1e-30))
        residual = (v - basis.T @ coeff).reshape(tgt["dW"].shape)         # ΔW residual
        # disjoint-content overlap (GATE 3): target digits ∩ population digits
        pop_digits = set(d for c in others for d in tuple(c["task"]))
        overlap = len(set(t_task) & pop_digits)
        # retrieval pool: target private images (label 1) + distractors = other tasks' private images (label 0)
        distract = np.concatenate([c["priv"] for c in others[:20]], axis=0)
        pool = np.concatenate([tgt["priv"], distract], axis=0)
        labels = np.concatenate([np.ones(len(tgt["priv"])), np.zeros(len(distract))])
        auc_raw = retrieval_auc(tgt["dW"], pool, labels, mu)
        auc_res = retrieval_auc(residual, pool, labels, mu)
        rand = RNG.standard_normal(tgt["dW"].shape); rand *= np.linalg.norm(tgt["dW"]) / np.linalg.norm(rand)
        auc_rand = retrieval_auc(rand, pool, labels, mu)                  # GATE 4 baseline
        per_target.append((t_task, auc_raw, auc_res, auc_res - auc_raw, auc_rand, proj_frac, proj_top, overlap))

    arr = per_target
    araw = np.array([r[1] for r in arr]); ares = np.array([r[2] for r in arr]); gain = np.array([r[3] for r in arr])
    arand = np.array([r[4] for r in arr]); pf = np.array([r[5] for r in arr]); ptop = np.array([r[6] for r in arr])
    ov = np.array([r[7] for r in arr])

    print("\n=== GATE 3 — disjoint-content overlap (target digits ∩ LOO-population digits) ===")
    print(f"  overlap per target: max={ov.max()} mean={ov.mean():.2f}  → {'DISJOINT ✓ (shared subspace = θ0 common-mode)' if ov.max()==0 else 'NON-DISJOINT ✗ (subspace absorbs target signal)'}")
    print("\n=== GATE 1 — PROJECTION test (target-ΔW energy in the LOO-shared subspace; must be LOW/moderate) ===")
    print(f"  energy-fraction in shared subspace: mean={pf.mean():.3f}  (top-singular-dir: mean={ptop.mean():.3f})")
    print(f"  → {'private direction largely ⊥ shared (LOO honest)' if pf.mean()<0.5 else 'HIGH — shared absorbs target signal (subtraction may be circular)'}")
    print("\n=== GATE 2 — HEADROOM (single-adapter raw retrieval AUC must be mid-range) ===")
    print(f"  raw-adapter AUC: mean={araw.mean():.3f} [{araw.min():.3f},{araw.max():.3f}]  random-adapter baseline={arand.mean():.3f}")
    mid = 0.55 < araw.mean() < 0.95
    print(f"  → {'MID-RANGE ✓ (room for a gain)' if mid else 'saturated/floor — gain uninterpretable'}")

    # GAIN with cluster-robust CI over the 5 tasks (GATE 4 = vs random-adapter, already ~0.5)
    utasks = sorted(set(r[0] for r in arr))
    cl = np.array([gain[[i for i, r in enumerate(arr) if r[0] == t]].mean() for t in utasks])
    G = len(cl); est = float(cl.mean()); se = cl.std(ddof=1) / np.sqrt(G) if G > 1 else float("nan")
    t = stats.t.ppf(0.975, G - 1) if G > 1 else float("nan")
    ci = (est - t * se, est + t * se)
    print("\n=== THE BALLGAME — GAIN = AUC(LOO-residual) − AUC(raw adapter) [vs random-adapter baseline ~0.5] ===")
    print(f"  AUC raw={araw.mean():.3f}  residual={ares.mean():.3f}  random={arand.mean():.3f}")
    print(f"  GAIN = {est:+.3f}  CI95 [{ci[0]:+.3f}, {ci[1]:+.3f}]  (G={G} task-clusters)")
    verdict = ("ECOSYSTEM EFFECT (CI excludes 0, positive)" if ci[0] > 0 else
               "NEGATIVE — subtraction HURTS (CI excludes 0, negative)" if ci[1] < 0 else
               "NULL — CI spans 0 (no population amplification detected)")
    print(f"  → {verdict}")
    print("\n  [observe-framed | population(>weakest)-attacker | not a 'confirmation' — a first honest read]")


if __name__ == "__main__":
    main()
