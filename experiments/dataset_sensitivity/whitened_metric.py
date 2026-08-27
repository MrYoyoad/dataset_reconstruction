"""Whitened (Mahalanobis) dataset-sensitivity estimator — ONE metric for the whole program.

Implements the protocol in ``notes/whitened_sensitivity_metric.md`` (sections
"PRIMARY READOUT: permutation null", "Estimation", and the "CORRECTION (arm-B
post-mortem)" that mandates a **3-WAY disjoint cross-fit**). The question every
dataset-sensitivity arm asks is a *detection* problem: how distinguishable is
dataset D from D' by looking at the adapter ΔW=BA, given that training-seed
randomness already jiggles it? The canonical answer is the noise-whitened mean
effect

    d²(D, D') = Δμᵀ Σ⁻¹ Δμ ,   Δμ = μ(D') − μ(D),   Σ = seed-noise covariance,

reported NOT as the raw (upward-biased) plug-in d̂², but as a permutation-null
effect size. This module is pure (tensors in, dict out) so it can be reused
across the whole ablation battery (arm B and the rest).

THE THREE PROTOCOL PIECES (all mandatory, do not conflate — they solve different
problems):

  1. PERMUTATION NULL  — THE de-biaser.  The null distribution of d̂² under
     SIGN-FLIPS of the K paired diffs (v_j → s_j·v_j, s_j∈{±1}).  The null carries
     the IDENTICAL estimator bias (same K, p, folds, ρ; only the labels flipped),
     so ``sensitivity = d̂²_obs − mean(d̂²_null)`` removes it by construction, at
     ANY K, with NO Gaussian / equal-Σ assumption.  ``pvalue`` = fraction of null
     draws ≥ observed.

  2. 3-WAY DISJOINT CROSS-FIT — anti-circularity, NOT the de-biaser.  The metric
     couples three quantities: the signal subspace U, the numerator Δμ·U, and the
     denominator λ (noise variance along U).  A 2-WAY split (U + λ from ONE fold,
     only Δμ cross-fit) is INSUFFICIENT and PROVABLY WRONG: λ is then measured
     along a subspace its own samples helped define, so it is selection-biased
     small (winner's curse) and d̂² INFLATES rather than converging in K (arm-B
     post-mortem: d²(N=64) went 63→161 as K 50→100 — a real quantity converges).
     The fix rotates over THREE DISJOINT roles:
        * role A → defines U = top-p right singular vectors of fold-A paired diffs.
        * role B → numerator: Δμ_B = mean of fold-B diffs; num_i = (Δμ_B·u_i)².
        * role C → denominator: λ_i = variance of fold-C RESEED adapters along u_i.
     U, numerator, and denominator each come from DISJOINT folds; λ is NEVER
     measured along a subspace its own samples helped define.  Rotate the three
     roles over all ordered fold-triples and average (double-ML) to keep power.

  3. SHRINKAGE ρ — the primary regularizer.  Whiten by (λ_i + ρ·λ_ref)⁻¹ per
     direction; ρ is chosen from ``shrink_list`` by CV against the null (the ρ
     maximizing the separation d̂²_obs − mean(d̂²_null)).

The whitening subspace is **signal-defined** (top-p directions of the fold-A
paired-diffs), never Σ's own col-space — restricting to the top-p *noise*
directions would discard exactly the low-noise directions where a targeted change
is most detectable.  Everything runs in flattened ΔW=BA vector space (the tensors
are already ΔW; never touch raw B/A) and in float64.

Gates reported alongside (the four detector equivalences hold only under
equal-Σ Gaussian; the permutation null survives if they fail):
  * Gaussianity of the seed noise — skew + excess kurtosis of the reseed
    projections onto the top-p subspace.
  * eff_rank(Σ) — participation-ratio effective rank of the seed-noise covariance.
"""

from __future__ import annotations

import contextlib
import itertools
from typing import List, Sequence

import numpy as np
import torch

__all__ = ["whitened_sensitivity"]


@contextlib.contextmanager
def _thread_guard(cap: int = 4):
    """Temporarily cap torch CPU threads, then restore the caller's setting.

    The estimator does many *tiny* float64 linear-algebra ops. On a fat node
    (WEXAC GPU nodes expose 256 CPUs) the default thread pool oversubscribes and
    these microsecond ops thrash for minutes. Capping to a handful of threads
    turns the whole estimator into a fraction of a second; the caller's global
    thread count is restored on exit so this stays a pure, side-effect-free call.
    """
    prev = torch.get_num_threads()
    try:
        torch.set_num_threads(max(1, min(prev, cap)))
        yield
    finally:
        torch.set_num_threads(prev)


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
def _stack_flat(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Flatten each [out, in] tensor to a vector and stack -> [K, D] float64."""
    rows = [t.reshape(-1).to(torch.float64) for t in tensors]
    return torch.stack(rows, dim=0)


def _eig_from_centered(N: torch.Tensor):
    """Eigenpairs of Σ = (1/(m-1)) Nᵀ N from centered samples N [m, D].

    Returns (eigvals [r] descending, Vt [r, D] right singular vectors as rows).
    Never forms the D×D covariance — works through the thin SVD of N.
    """
    m = N.shape[0]
    # full_matrices=False -> U [m, m], S [m], Vt [m, D]
    _, S, Vt = torch.linalg.svd(N, full_matrices=False)
    eigvals = (S.pow(2) / max(m - 1, 1)).clamp_min(0.0)
    return eigvals, Vt


def _signal_subspace(V_role_a: torch.Tensor, p: int) -> torch.Tensor:
    """Top-p SIGNAL directions: right singular vectors of the (uncentered) role-A
    paired-diffs V_role_a [na, D]. Returned as rows [p, D].

    Uncentered SVD ⇒ the second-moment matrix Vᵀ V is sign-flip invariant, so the
    subspace is a fixed nuisance under the permutation null (only the coherent
    amplitude Δμ̂ changes) — exactly what makes the null exact by construction.
    """
    _, _, Vt = torch.linalg.svd(V_role_a, full_matrices=False)
    return Vt[:p]                                    # [p, D]


def _diag_var_along(R_fold: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Per-direction centered variance of reseeds R_fold [m, D] along rows of
    U [p, D]. Returns λ [p]. This is the denominator's noise variance measured on
    a fold DISJOINT from the one that defined U (role C ⟂ role A)."""
    N = R_fold - R_fold.mean(dim=0, keepdim=True)     # [m, D] centered
    P = N @ U.t()                                     # [m, p]
    m = P.shape[0]
    return P.pow(2).sum(dim=0) / max(m - 1, 1)        # [p]


def _participation_eff_rank(eigvals: torch.Tensor) -> float:
    """Participation-ratio effective rank: (Σλ)² / Σλ²."""
    s1 = float(eigvals.sum())
    s2 = float(eigvals.pow(2).sum())
    return (s1 * s1 / s2) if s2 > 0 else 0.0


def _sign_draws(K: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Null sign vectors in {±1}^K, EXCLUDING the trivial ±all-ones (the observed
    labelling and its global-sign mirror). Enumerate all 2^K when K is small
    (≤18), else draw n_perm random vectors.
    """
    if K <= 18:
        allc = np.array(list(itertools.product((1.0, -1.0), repeat=K)),
                        dtype=np.float64)                       # [2^K, K]
        keep = ~(np.all(allc == 1.0, axis=1) | np.all(allc == -1.0, axis=1))
        return allc[keep]
    S = rng.choice(np.array([1.0, -1.0]), size=(n_perm, K)).astype(np.float64)
    # scrub any accidental all-ones / all-minus-ones draw
    bad = np.all(S == S[:, :1], axis=1)
    if bad.any():
        S[bad, 0] = -S[bad, 0]
    return S


# ----------------------------------------------------------------------------- #
# main estimator
# ----------------------------------------------------------------------------- #
def whitened_sensitivity(v_list: List[torch.Tensor],
                         reseed_list: List[torch.Tensor],
                         n_folds: int = 5,
                         p_max: int = 3,
                         shrink_list: Sequence[float] = (1e-3, 1e-2, 1e-1),
                         n_perm: int = 1000,
                         seed: int = 0) -> dict:
    """Permutation-null whitened sensitivity of a data change D -> D' (thread-guarded).

    Thin wrapper that caps CPU threads (fat-node oversubscription guard) around
    the pure implementation; the caller's thread setting is restored on return.
    See ``_whitened_sensitivity_impl`` for the full contract.
    """
    with _thread_guard():
        return _whitened_sensitivity_impl(
            v_list, reseed_list, n_folds=n_folds, p_max=p_max,
            shrink_list=shrink_list, n_perm=n_perm, seed=seed)


def _whitened_sensitivity_impl(v_list: List[torch.Tensor],
                               reseed_list: List[torch.Tensor],
                               n_folds: int = 5,
                               p_max: int = 3,
                               shrink_list: Sequence[float] = (1e-3, 1e-2, 1e-1),
                               n_perm: int = 1000,
                               seed: int = 0) -> dict:
    """Permutation-null whitened sensitivity of a data change D -> D', 3-way cross-fit.

    Parameters
    ----------
    v_list : list of K tensors, each the paired-per-seed diff
        v_j = ΔW(D, seed_j) − ΔW(D', seed_j), shape [out, in]. Flattened
        internally to vectors of length D = out*in.
    reseed_list : list of K tensors, each a reseed adapter ΔW(D, seed_j),
        shape [out, in] — the noise ensemble whose covariance is Σ.
    n_folds : number of disjoint folds (≥3). The three roles (subspace U,
        numerator Δμ·U, denominator λ) are assigned to distinct folds and the
        assignment is rotated over all ordered fold-triples and averaged.
    p_max : cap on the whitening-subspace dimension p (kept ≪ per-fold size).
    shrink_list : candidate relative shrinkages ρ (whiten by (λ_i + ρ·λ_ref)⁻¹);
        ρ is selected by CV against the null.
    n_perm : number of random sign-flip null draws (all 2^K enumerated if K≤18).
    seed : RNG seed for fold assignment and sign draws.

    Returns
    -------
    dict with keys: d2_obs, d2_null_mean, sensitivity, pvalue,
        whitened_spectrum, qeff_count, sigma_eff_rank, gaussianity_skew,
        gaussianity_kurt, rho_selected, n_folds, p_used.
    """
    if len(v_list) != len(reseed_list):
        raise ValueError("v_list and reseed_list must have equal length K "
                         f"(got {len(v_list)} and {len(reseed_list)})")
    if n_folds < 3:
        raise ValueError(f"n_folds={n_folds} too small for the 3-WAY disjoint "
                         "cross-fit (need at least 3 folds for roles A/B/C)")
    V = _stack_flat(v_list)                          # [K, D]
    R = _stack_flat(reseed_list)                     # [K, D]
    K, D = V.shape
    if K < 2 * n_folds:
        raise ValueError(f"K={K} too small for n_folds={n_folds} "
                         "(need at least 2 seeds per fold)")

    rng = np.random.default_rng(seed)

    # --- fold assignment (shuffled, balanced) -------------------------------- #
    perm = rng.permutation(K)
    folds = [np.sort(f) for f in np.array_split(perm, n_folds)]
    min_fold = min(len(f) for f in folds)

    # freeze p across triples: p ≪ per-fold size, and ≤ role-A rank & ambient dim
    p = int(max(1, min(p_max, min_fold - 1, D)))

    # ------------------------------------------------------------------------- #
    # Per-triple 3-WAY cross-fit precomputation (independent of ρ and the signs).
    #   role A (fa) → U  = top-p right singular vectors of fold-A paired diffs.
    #   role B (fb) → WB = fold-B diffs projected on U (numerator amplitudes; the
    #                 signed mean under the null is (s_B·WB)/nb).
    #   role C (fc) → λ  = per-direction reseed variance along U (denominator),
    #                 with λ_ref = dominant fold-C seed-noise scale for ρ.
    # Every role reads a DISJOINT fold ⇒ λ is never measured along a subspace its
    # own samples helped define. Rotate over ALL ordered triples and average.
    # ------------------------------------------------------------------------- #
    triple_cache = []
    for fa, fb, fc in itertools.permutations(range(n_folds), 3):
        idx_a, idx_b, idx_c = folds[fa], folds[fb], folds[fc]
        U = _signal_subspace(V[idx_a], p)             # [p, D] role-A subspace
        WB = V[idx_b] @ U.t()                         # [nb, p] role-B amplitudes
        lam = _diag_var_along(R[idx_c], U)            # [p] role-C denominator var
        # ρ reference: dominant seed-noise scale of the DISJOINT role-C fold.
        N_c = R[idx_c] - R[idx_c].mean(dim=0, keepdim=True)
        eig_c, _ = _eig_from_centered(N_c)
        lam_ref = float(eig_c[0]) if eig_c.numel() else 0.0
        lam_ref = max(lam_ref, 1e-30)
        triple_cache.append(dict(WB=WB, lam=lam, lam_ref=lam_ref, idx_b=idx_b))

    def _d2(signs: np.ndarray, rho: float) -> np.ndarray:
        """d²(rho) for a batch of sign vectors `signs` [P, K]. Averaged over the
        ordered fold-triples. Per direction: num_i / (λ_i + ρ·λ_ref)."""
        P = signs.shape[0]
        s_t = torch.from_numpy(signs)                 # [P, K] float64
        acc = torch.zeros(P, dtype=torch.float64)
        for tc in triple_cache:
            idx_b = tc["idx_b"]
            nb = len(idx_b)
            s_b = s_t[:, idx_b]                        # [P, nb]
            c = (s_b @ tc["WB"]) / nb                  # [P, p] signed-mean amplitude
            denom = tc["lam"] + rho * tc["lam_ref"]    # [p] shrinkage-regularized λ
            acc += (c.pow(2) / denom).sum(dim=1)       # Σ_i num_i / denom_i
        return (acc / len(triple_cache)).numpy()

    ones = np.ones((1, K), dtype=np.float64)          # observed labelling
    null_signs = _sign_draws(K, n_perm, rng)          # [P, K]

    # ------------------------------------------------------------------------- #
    # Shrinkage ρ selection: CV against the null. Pick ρ maximizing the
    # separation d²_obs − mean(d²_null). Same sign draws reused across ρ.
    # ------------------------------------------------------------------------- #
    best = None
    for rho in shrink_list:
        d2_obs = float(_d2(ones, rho)[0])
        d2_null = _d2(null_signs, rho)                # [P]
        null_mean = float(d2_null.mean())
        sep = d2_obs - null_mean
        if best is None or sep > best["sep"]:
            best = dict(rho=float(rho), d2_obs=d2_obs, null_mean=null_mean,
                        sep=sep, d2_null=d2_null)

    d2_obs = best["d2_obs"]
    d2_null = best["d2_null"]
    null_mean = best["null_mean"]
    rho_sel = best["rho"]
    # add-one (include the observed in the reference set) -> honest, never exactly 0
    n_ge = int((d2_null >= d2_obs).sum())
    pvalue = (n_ge + 1) / (d2_null.shape[0] + 1)

    # ------------------------------------------------------------------------- #
    # DIAGNOSTICS on the FULL data (Δμ̂ from all K, Σ from all K reseeds):
    #   whitened spectrum, q_eff count, eff_rank(Σ), Gaussianity gate.
    # These are descriptive gates, not the headline magnitude (which is the
    # 3-way d2_obs/sensitivity above); they use the full sample for stability.
    # ------------------------------------------------------------------------- #
    dmu = V.mean(dim=0)                                # [D] full Δμ̂
    Up_full = _signal_subspace(V, p)                   # [p, D]
    N_full = R - R.mean(dim=0, keepdim=True)           # [K, D] centered reseed
    m_full = N_full.shape[0]

    eig_full, _ = _eig_from_centered(N_full)
    sigma_eff_rank = _participation_eff_rank(eig_full)
    lam_max_full = float(eig_full[0]) if eig_full.numel() else 1e-30
    rho_abs_full = rho_sel * max(lam_max_full, 1e-30)

    proj_noise = N_full @ Up_full.t()                  # [K, p] reseed on subspace
    var_k = proj_noise.pow(2).sum(dim=0) / max(m_full - 1, 1)   # [p] noise var / dir
    sig_k = (Up_full @ dmu).abs()                      # [p] signal per direction
    # per-direction whitened signal = |Δμ̂·u_k| / noise_std_k (shrinkage-floored)
    noise_std = (var_k + rho_abs_full).sqrt()
    whitened_spectrum = (sig_k / noise_std)            # [p]
    qeff_count = int((whitened_spectrum > 1.0).sum())

    # Gaussianity of the seed noise on the top-p subspace (standardize each
    # direction, pool, then skew + excess kurtosis).
    std_k = proj_noise.std(dim=0, unbiased=True).clamp_min(1e-30)
    z = (proj_noise / std_k).reshape(-1)               # pooled standardized proj
    z = z - z.mean()
    zstd = z.std(unbiased=True).clamp_min(1e-30)
    z = z / zstd
    gaussianity_skew = float((z.pow(3)).mean())
    gaussianity_kurt = float((z.pow(4)).mean() - 3.0)

    return dict(
        d2_obs=d2_obs,
        d2_null_mean=null_mean,
        sensitivity=d2_obs - null_mean,
        pvalue=pvalue,
        whitened_spectrum=[float(x) for x in whitened_spectrum.tolist()],
        qeff_count=qeff_count,
        sigma_eff_rank=sigma_eff_rank,
        gaussianity_skew=gaussianity_skew,
        gaussianity_kurt=gaussianity_kurt,
        rho_selected=rho_sel,
        n_folds=n_folds,
        p_used=p,
    )


# ----------------------------------------------------------------------------- #
# Superseded 2-WAY estimator — kept ONLY for the K-convergence regression gate.
# U AND λ come from the SAME (train) split; only the numerator is cross-fit. This
# is the proven-wrong estimator (λ selection-biased small ⇒ d² inflates and does
# NOT converge in K). The self-test computes it inline to prove the 3-way fix
# matters; it is NOT part of the public API and must never be used to report a
# magnitude.
# ----------------------------------------------------------------------------- #
def _d2_obs_2way(V: torch.Tensor, R: torch.Tensor, n_folds: int, p: int,
                 rho: float, seed: int) -> float:
    """Observed plug-in d² under the OLD 2-way split (U+λ share the train fold)."""
    K = V.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(K)
    folds = [np.sort(f) for f in np.array_split(perm, n_folds)]
    acc = 0.0
    for f in folds:
        test_mask = np.zeros(K, dtype=bool)
        test_mask[f] = True
        train_mask = ~test_mask
        U = _signal_subspace(V[train_mask], p)         # subspace from TRAIN
        lam = _diag_var_along(R[train_mask], U)        # λ from the SAME TRAIN split
        N_c = R[train_mask] - R[train_mask].mean(dim=0, keepdim=True)
        eig_c, _ = _eig_from_centered(N_c)
        lam_ref = max(float(eig_c[0]) if eig_c.numel() else 0.0, 1e-30)
        num = (V[test_mask].mean(dim=0) @ U.t()).pow(2)   # numerator cross-fit only
        acc += float((num / (lam + rho * lam_ref)).sum())
    return acc / len(folds)


def _d2_obs_3way(V: torch.Tensor, R: torch.Tensor, n_folds: int, p: int,
                 rho: float, seed: int) -> float:
    """Observed plug-in d² under the 3-WAY disjoint split, at a FIXED ρ (no null,
    no ρ-selection). Mirrors ``_d2_obs_2way`` term-for-term so the K-convergence
    gate isolates the ONLY difference — where λ is measured (disjoint fold-C here
    vs the U-defining split in the 2-way). This is the same math the public
    estimator uses per triple; kept standalone so the gate can hold ρ/p/folds
    identical across the two estimators."""
    K = V.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(K)
    folds = [np.sort(f) for f in np.array_split(perm, n_folds)]
    tot = 0.0
    cnt = 0
    for fa, fb, fc in itertools.permutations(range(n_folds), 3):
        U = _signal_subspace(V[folds[fa]], p)          # role A: subspace
        lam = _diag_var_along(R[folds[fc]], U)         # role C: λ (DISJOINT fold)
        N_c = R[folds[fc]] - R[folds[fc]].mean(dim=0, keepdim=True)
        eig_c, _ = _eig_from_centered(N_c)
        lam_ref = max(float(eig_c[0]) if eig_c.numel() else 0.0, 1e-30)
        num = (V[folds[fb]].mean(dim=0) @ U.t()).pow(2)   # role B: numerator
        tot += float((num / (lam + rho * lam_ref)).sum())
        cnt += 1
    return tot / cnt


# ----------------------------------------------------------------------------- #
# ACCEPTANCE-GATE self-test (synthetic CPU data, no GPU needed)
# ----------------------------------------------------------------------------- #
def _selftest() -> bool:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    f64 = torch.float64

    D, K = 50, 40
    out_shape = (5, 10)                                # [out, in], out*in = D

    # Orthonormal basis; dominant seed-noise lives in the first 3 directions.
    A = torch.randn(D, D, generator=g, dtype=f64)
    Q, _ = torch.linalg.qr(A)                          # columns orthonormal [D, D]
    noise_std = torch.tensor([1.0, 0.6, 0.4], dtype=f64)   # 3 dominant modes
    iso = 0.02                                         # tiny isotropic floor

    def draw_noise(n):
        """n seed-noise vectors [n, D] from the low-rank + isotropic model."""
        coeff = torch.randn(n, 3, generator=g, dtype=f64) * noise_std     # [n,3]
        low = coeff @ Q[:, :3].t()                                        # [n, D]
        return low + iso * torch.randn(n, D, generator=g, dtype=f64)

    def as_tensors(mat):                               # rows [K, D] -> list of [out,in]
        return [mat[i].reshape(out_shape).clone() for i in range(mat.shape[0])]

    base = torch.randn(D, generator=g, dtype=f64)      # arbitrary D-mean (cancels)
    reseed_mat = base.unsqueeze(0) + draw_noise(K)     # ΔW(D, seed_j)
    reseed = as_tensors(reseed_mat)

    u_orth = Q[:, 10]                                  # orthogonal to the noise modes
    u_dom = Q[:, 0]                                    # the dominant noise direction

    results = {}

    # (a) signal ORTHOGONAL to the (low-rank) noise -> detectable.
    dmu_a = 0.5 * u_orth
    v_a = as_tensors(dmu_a.unsqueeze(0) + draw_noise(K))
    results["a"] = whitened_sensitivity(v_a, reseed, n_perm=2000, seed=1)

    # (b) signal ALIGNED with the dominant noise direction, comparable to the
    #     seed-noise-driven displacement of the mean estimate -> masked.
    dmu_b = 0.15 * u_dom
    v_b = as_tensors(dmu_b.unsqueeze(0) + draw_noise(K))
    results["b"] = whitened_sensitivity(v_b, reseed, n_perm=2000, seed=2)

    # (c) pure-noise NULL: v_j = reseed'_j − reseed''_j (no injected signal).
    v_c = as_tensors(draw_noise(K) - draw_noise(K))
    results["c"] = whitened_sensitivity(v_c, reseed, n_perm=2000, seed=3)

    checks = {
        "a": (results["a"]["sensitivity"] > 0 and results["a"]["pvalue"] < 0.05),
        "b": (abs(results["b"]["sensitivity"]) < 0.5 * max(results["a"]["sensitivity"], 1e-9)
              and results["b"]["pvalue"] > 0.10),
        "c": (results["c"]["pvalue"] >= 0.10
              and abs(results["c"]["sensitivity"]) < 0.5 * max(results["a"]["sensitivity"], 1e-9)),
    }

    print("=" * 74)
    print("whitened_sensitivity — acceptance-gate self-test (D=50, K=40)")
    print("=" * 74)
    labels = {
        "a": "(a) signal ORTHOGONAL to low-rank noise   -> expect DETECTABLE",
        "b": "(b) signal ALIGNED w/ dominant noise dir   -> expect MASKED",
        "c": "(c) pure-noise NULL (reseed differences)   -> expect NULL",
    }
    all_pass = True
    for key in ("a", "b", "c"):
        r = results[key]
        ok = checks[key]
        all_pass = all_pass and ok
        print(f"\n{labels[key]}")
        print(f"    d2_obs={r['d2_obs']:.4g}  null_mean={r['d2_null_mean']:.4g}  "
              f"sensitivity={r['sensitivity']:.4g}  pvalue={r['pvalue']:.4g}")
        print(f"    rho_selected={r['rho_selected']:.0e}  p_used={r['p_used']}  "
              f"qeff_count={r['qeff_count']}  eff_rank(Σ)={r['sigma_eff_rank']:.2f}")
        print(f"    whitened_spectrum={[round(x, 3) for x in r['whitened_spectrum']]}  "
              f"gauss(skew={r['gaussianity_skew']:+.2f}, exkurt={r['gaussianity_kurt']:+.2f})")
        print(f"    -> {'PASS' if ok else 'FAIL'}")

    # ------------------------------------------------------------------------- #
    # (d) K-CONVERGENCE gate — the arm-B post-mortem requirement.
    #
    # Convergence is a statement about the ESTIMATOR'S EXPECTATION: E[d²(K)] must
    # stabilize as K→2K.  We estimate it as the mean over many INDEPENDENT
    # synthetic datasets (the sampling analogue of the expectation) so the gate
    # reflects the estimator's bias, not one lucky noise draw.
    #
    # The signal sits in a genuinely LOW-noise reseed direction (var 0.03) hidden
    # among higher-noise nuisance directions (var 1.0), and carries HIGH diff
    # noise (var 1.0) so the subspace only marginally resolves it — the regime
    # where the winner's curse bites.  A correct estimator converges; the OLD
    # 2-way (U + λ from the SAME split) does NOT: with its large train fold it
    # keeps sharpening U onto the low-noise direction and dividing by a λ measured
    # on those very samples, so E[d²] climbs with K instead of settling.
    #
    # Assert (SAME data, SAME ρ/p/folds — the ONLY difference is disjointness):
    #   * 3-WAY  E[d²] STABLE:  |drift(K→2K)| ≤ 15%.
    #   * OLD 2-way INFLATES:   2K materially larger (≥ 25%, and ≥ 3× the 3-way
    #                           drift) — proving the fix matters.
    # ------------------------------------------------------------------------- #
    def gen_lownoise(Kn, seed):
        gg = torch.Generator().manual_seed(seed)
        Dn = 60
        Qn, _ = torch.linalg.qr(torch.randn(Dn, Dn, generator=gg, dtype=f64))
        res_var = torch.full((Dn,), 0.05, dtype=f64)   # bulk reseed noise
        res_var[0] = 0.03                              # signal dir: LOW reseed noise
        res_var[1:6] = 1.0                             # high-noise nuisance dirs
        diff_var = torch.full((Dn,), 0.05, dtype=f64)
        diff_var[0] = 1.0                              # signal dir: HIGH diff noise
        diff_var[1:6] = 1.0
        def draw(n, var):
            z = torch.randn(n, Dn, generator=gg, dtype=f64)
            return (z * var.sqrt()) @ Qn.t()           # noise in the Q basis
        base_n = torch.randn(Dn, generator=gg, dtype=f64)
        reseed_n = base_n.unsqueeze(0) + draw(Kn, res_var)
        dmu_n = 1.2 * Qn[:, 0]                          # signal along low-noise dir 0
        v_n = dmu_n.unsqueeze(0) + draw(Kn, diff_var)
        vt = [v_n[i].reshape(6, 10).clone() for i in range(Kn)]
        rt = [reseed_n[i].reshape(6, 10).clone() for i in range(Kn)]
        return vt, rt

    Kc, n_data, rho_c, p_c, nf_c = 30, 60, 1e-2, 3, 5

    def _epop(estimator, Kn):
        """E[d²] over n_data independent datasets at fixed ρ/p/folds."""
        vals = []
        for gs in range(n_data):
            vt, rt = gen_lownoise(Kn, gs)
            V_ = _stack_flat(vt); R_ = _stack_flat(rt)
            vals.append(estimator(V_, R_, nf_c, p_c, rho_c, 0))
        return float(np.mean(vals))

    with _thread_guard():
        new_k = _epop(_d2_obs_3way, Kc)
        new_2k = _epop(_d2_obs_3way, 2 * Kc)
        old_k = _epop(_d2_obs_2way, Kc)
        old_2k = _epop(_d2_obs_2way, 2 * Kc)
    new_drift = (new_2k - new_k) / max(new_k, 1e-9)
    old_infl = (old_2k - old_k) / max(old_k, 1e-9)

    # Also show the PUBLIC API on one representative dataset: the permutation
    # p-value is the K-stable primary readout (magnitude is trustworthy only once
    # the 3-way split passes this convergence gate).
    v_rep, r_rep = gen_lownoise(2 * Kc, 0)
    rep = whitened_sensitivity(v_rep, r_rep, n_folds=nf_c, n_perm=1500, seed=7)

    conv_ok = (abs(new_drift) <= 0.15
               and old_infl >= 0.25
               and old_infl >= 3.0 * abs(new_drift))

    print("\n" + "-" * 74)
    print(f"(d) K-CONVERGENCE gate  (signal in a low-reseed-noise direction; "
          f"E[d²] over {n_data} datasets)")
    print("-" * 74)
    print(f"    3-WAY disjoint  E[d²]:  K={Kc} -> {new_k:.4g}   2K={2*Kc} -> {new_2k:.4g}"
          f"   drift={100*new_drift:+.1f}%   (require |drift| <= 15%)")
    print(f"    OLD 2-way       E[d²]:  K={Kc} -> {old_k:.4g}   2K={2*Kc} -> {old_2k:.4g}"
          f"   inflation={100*old_infl:+.1f}%   (require >= 25% and >= 3x 3-way)")
    print(f"    public-API readout (rep dataset): pvalue={rep['pvalue']:.4g}  "
          f"d2_obs={rep['d2_obs']:.4g}  rho={rep['rho_selected']:.0e}")
    print(f"    -> 3-way STABLE: {abs(new_drift) <= 0.15}   |   OLD 2-way INFLATES: "
          f"{old_infl >= 0.25 and old_infl >= 3.0 * abs(new_drift)}")
    print(f"    -> {'PASS' if conv_ok else 'FAIL'}")

    all_pass = all_pass and conv_ok

    print("\n" + "=" * 74)
    n_pass = sum(checks.values()) + int(conv_ok)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'} ({n_pass}/4 cases)")
    print("=" * 74)
    return all_pass


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
