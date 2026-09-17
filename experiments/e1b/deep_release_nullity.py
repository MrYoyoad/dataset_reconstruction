#!/usr/bin/env python3
"""The RELEASE-ROUTE nullity column for yoado-83's multilayer k-sweep, on their exact join keys.

Their harness constructs the ZERO-DRIFT certificate directly and has no trained release, so it cannot compute this
column: doing so would mean adding a release-training path, which introduces drift and moves the regime their
rank-law test lives in. So this is option (b) -- a separate job joined on (checkpoint, image-index, k).

TWO COLUMNS, TWO ZERO SETS, NEVER DIFFERENCED. Their `cert_route_nullity` counts chart directions the stacked
certificate leaves unconstrained; this counts the family of (chart coords, seed) reproducing a trained release. The
containment runs one way, `{truth} subset {replay = 0} subset {C h = 0}`, so theirs can be large where this is
zero and that is the nesting rather than a disagreement. a8's capacity cap `k < m + r - N` is about THIS one.

CARRIED CAVEAT: this column requires a trained release, so it carries DRIFT by construction and is not a
zero-drift measurement. The two columns describe different regimes on the same axis.

DECISIONS THAT KEEP IT LIKE-FOR-LIKE with the affine law (which predicts `N*max(0, k - (m+r-N-1))`):
  * HEAD-ONLY adaptation, so depth enters through `phi` rather than through several adapted layers. A
    multi-layer-adapted release is a different cell and would be labelled one.
  * `T` FIXED across the whole sweep. Nullity does not depend on T, but condition number does, so a
    condition-number column swept at varying T would be meaningless.

FEASIBILITY, stated because it drives the choice of `r`. The Jacobian is `r(n+m) x (kN + rn)` with `n = 1000`, so
it grows with `r`: about 2 GB at r=16, 8 GB at r=32, and 31 GB at r=64 before any SVD workspace. r=16 puts the cap
at `11 + 16 - 8 - 1 = 18`, which places the transition between their k=16 and k=32 grid points -- visible inside
their own grid, which is what the column is for.

  python -m experiments.e1b.deep_release_nullity --r 16 --ks 16 32 66 96
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch

from experiments.exact_inversion.deep_stack import load_deep, inputs_of
from experiments.exact_inversion.trained_backbone import read_idx, PCAChart
from experiments.utils.identifiability import identifiability, assert_chart_contains, block_normalised_residual
from experiments.exact_inversion.lora_exact_inversion import git_hash

torch.set_default_dtype(torch.float64)


def log(s): print(s, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/exact_inversion/mnist_mlp_d15w1000.pth")
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=20); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 32, 66, 96])
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/e1b/deep_release_nullity.jsonl")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(os.path.dirname(a.out), exist_ok=True)

    Ws, b1, ck = load_deep(a.ckpt, dev)
    n = Ws[-1].shape[1]; m_base = Ws[-1].shape[0]; m = m_base + 1          # added class, 11th row zeroed
    cap = m + a.r - a.N - 1
    log(f"# JOIN KEY 1 checkpoint: {a.ckpt}  ({len(Ws)} weight matrices, head {tuple(Ws[-1].shape)})")

    # JOIN KEY 2 -- their exact recipe, reproduced verbatim
    Xte, yte = read_idx(a.data_root, "test")
    Xte_t = torch.tensor(Xte, device=dev)
    idx = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))[: a.N].to(dev)
    X_real = Xte_t[idx].T.contiguous()                                     # (784, N)
    log(f"# JOIN KEY 2 private image indices: {idx.tolist()}  labels {yte[idx.cpu().numpy()].tolist()}")

    Xtr, _ = read_idx(a.data_root, "train")
    Xtr_t = torch.tensor(Xtr[:50000], device=dev)

    phi = lambda X: inputs_of(X, Ws, b1)[-1]                               # the head's input; frozen backbone
    W0 = torch.cat([Ws[-1], torch.zeros(1, n, dtype=torch.float64, device=dev)], 0)
    y = torch.full((a.N,), m - 1, device=dev)
    Y = torch.eye(m, device=dev, dtype=torch.float64)[y].T
    A0 = (1.0 / math.sqrt(n) * torch.randn(a.r, n, dtype=torch.float64,
                                           generator=torch.Generator().manual_seed(a.seed + 7))).to(dev)
    log(f"# n={n} m={m} r={a.r} N={a.N} T={a.T}  ->  cap = m+r-N-1 = {cap}; predicted nullity = N*max(0,k-cap)")
    log(f"# HEAD-ONLY adaptation; release carries DRIFT by construction (not a zero-drift measurement)\n")

    def train_release(H):
        A, B = A0.clone(), torch.zeros(m, a.r, dtype=torch.float64, device=dev)
        for _ in range(a.T):
            z = W0 @ H + B @ (A @ H); D = (torch.softmax(z, 0) - Y) / a.N
            B, A = B - a.lr * (D @ (A @ H).T), A - a.lr * (B.T @ D @ H.T)
        return A, B

    log(f"{'k':>5} {'unknowns':>9} {'equations':>10} {'rank':>7} {'nullity':>8} {'predicted':>10} {'hit':>5} "
        f"{'GAP@cut':>10} {'cond':>10} {'chart_err':>10}  {'sec':>6}")
    log("# GAP@cut is the decisive column: a rank cut is only a rank cut if it sits in a spectral GAP. A smooth "
        "decay through the threshold means the 'nullity' is a cutoff choice, not a nullity.")
    rows = []
    for k in a.ks:
        t0 = time.time()
        chart = PCAChart(Xtr_t, k, dev)
        Won = chart.coords_of(X_real)
        X_on = chart.psi(Won)
        assert_chart_contains(chart.psi, Won, X_on)
        chart_err = float((torch.linalg.norm(X_on - X_real, dim=0) / torch.linalg.norm(X_real, dim=0)).median())
        H = phi(X_on)
        A_T, B_T = train_release(H)

        def sim(Wc, A0c):
            Hc = phi(chart.psi(Wc))
            A, B = A0c, torch.zeros(m, a.r, dtype=torch.float64, device=dev)
            for _ in range(a.T):
                z = W0 @ Hc + B @ (A @ Hc); D = (torch.softmax(z, 0) - Y) / a.N
                B, A = B - a.lr * (D @ (A @ Hc).T), A - a.lr * (B.T @ D @ Hc.T)
            return (B, A)

        res = block_normalised_residual(sim, (B_T, A_T))
        r_ = identifiability(res, (Won, A0), mode="rev", chunk_size=a.chunk)
        pred = a.N * max(0, k - cap)
        hit = (r_.nullity == pred)
        log(f"{k:>5} {r_.n_unknowns:>9} {r_.n_equations:>10} {r_.rank:>7} {r_.nullity:>8} {pred:>10} "
            f"{'HIT' if hit else 'MISS':>5} {r_.gap_at_cut:>10.3e} {r_.condition_number:>10.3e} "
            f"{chart_err:>10.4f}  {time.time()-t0:>6.0f}")
        sv = r_.singular_values
        lo, hi = max(0, r_.rank - 6), min(len(sv), r_.rank + 6)
        log(f"      spectrum around the cut (index {lo}..{hi-1}, cut after {r_.rank-1}): "
            + "  ".join(f"{'|' if i == r_.rank else ''}{sv[i]/sv[0]:.2e}" for i in range(lo, hi)))
        sv0 = sv[0]
        ladder = [(t, int(sum(1 for v in sv if v > t * sv0))) for t in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16)]
        log("      RANK vs THRESHOLD (a real rank is FLAT across decades; a sliding count means there is no rank): "
            + "  ".join(f"{t:.0e}->{rk}" for t, rk in ladder))
        spread = max(rk for _, rk in ladder) - min(rk for _, rk in ladder)
        log(f"      rank moves by {spread} across ten decades of threshold"
            + ("  <-- NO WELL-DEFINED RANK" if spread > 8 else "  <-- stable"))
        if r_.gap_at_cut < 1e3:
            log(f"      *** GAP AT CUT IS ONLY {r_.gap_at_cut:.1e} -- the spectrum decays smoothly through the "
                f"threshold, so this 'nullity' is NOT trustworthy as a rank statement ***")
        rows.append(dict(k=k, predicted=pred, hit=bool(hit), chart_error=chart_err,
                         release_route_nullity=r_.nullity, rank_vs_threshold=ladder,
                         rank_spread_over_decades=spread, **r_.as_dict()))
        with open(a.out, "a") as fh:
            fh.write(json.dumps(dict(part="deep_release_nullity", join_checkpoint=a.ckpt,
                                     join_image_indices=idx.tolist(), join_k=k, regime="TRAINED RELEASE (carries drift)",
                                     adaptation="head-only", n=n, m=m, r=a.r, N=a.N, T=a.T, cap=cap,
                                     precision="fp64", note="release-route nullity; NOT comparable with "
                                     "cert_route_nullity -- {truth} subset {replay=0} subset {Ch=0}, never difference",
                                     **rows[-1], git=git_hash(), host=socket.gethostname(),
                                     cmd=" ".join(sys.argv))) + "\n")
    log(f"\n# {sum(x['hit'] for x in rows)}/{len(rows)} cells match the affine law on a trained 15-layer backbone.")


if __name__ == "__main__":
    main()
