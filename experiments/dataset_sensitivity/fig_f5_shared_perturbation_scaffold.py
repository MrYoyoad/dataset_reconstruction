#!/usr/bin/env python
"""
FIGURE F5 — RECOVER THE SHARED PERTURBATION  (SCAFFOLD ONLY, COMPUTE-GATED).

######################################################################
#  HARD GATE: this script fires NO experiment by default.            #
#  The compute path (fine-tune + attack) refuses to run unless the   #
#  explicit --approved flag is passed. Default: print the gate       #
#  message and exit 0. Awaiting Gal's compute approval.              #
######################################################################

THREAT MODEL (a DIFFERENT, EASIER target than per-image reconstruction):
Apply a fixed SHARED transform T_theta (a common rotation theta, or a common
blur sigma) to ALL N training images, fine-tune the LoRA adapter, then attack
to recover the transform's scalar PARAMETER (theta or sigma) — NOT the images.
Readout: recovery error of the shared-transform parameter vs its true value,
across transform strength.

WHY THIS IS AN EXTENSION, NOT A CONTRADICTION (framing, mandatory):
Recovering ONE scalar shared by N images is a much weaker target than
reconstructing the per-image pixels that the 0/40 weakest-attacker baseline
measures. A recovery SKILL > 0 here is STRUCTURAL leakage — an extension PAST
the weakest-attacker valley (prior-free, adapter-only, per-image), NOT a
contradiction of it.

This scaffold REUSES the similarity-ladder rot/blur harness (_rotate,
_gauss_blur) so the shared transform is byte-identical to the per-image rungs.
The metric block (recovery_error / trivial baseline / skill / bootstrap CI) is
implemented per the yoado-6d spec; the fine-tune+attack PIPELINE is a stub that
the executer wires only after Gal approves compute.
"""
import os
import sys
import math
import argparse

import torch

# NOTE: the shared-transform harness (_rotate, _gauss_blur) is imported LAZILY
# inside apply_shared_transform (below) so this scaffold's gate + metric functions
# stay importable without the full fine-tune stack (torchvision/timm live only on
# WEXAC). The compute path reuses the EXACT ladder harness so the common transform
# matches the per-image rungs byte-for-byte.

FIGURES = "/home/projects/galvardi/yoado/figures/shared_perturbation"

CAPTION = (
    "F5 - Recovering the shared transform is a DIFFERENT, EASIER target than the "
    "per-image pixels the 0/40 measures - a structural-leakage EXTENSION past the "
    "weakest-attacker valley, not a contradiction.  (recovers ONE scalar from N "
    "images; skill>0 = STRUCTURAL leakage, an extension PAST the weakest-attacker "
    "0/40, not a contradiction.)"
)


# --------------------------------------------------------------------------- #
# SHARED TRANSFORM — apply a SINGLE parameter to ALL N images (not per-image). #
# --------------------------------------------------------------------------- #
def apply_shared_transform(images, kind, param):
    """images: [N,1,28,28] float in [0,1]. Apply the SAME transform (kind, param)
    to every image. kind in {'rot' (deg), 'blur' (sigma)}. Reuses the ladder harness."""
    # Lazy import: keeps the gate + metric functions usable without the WEXAC-only
    # fine-tune stack; this is only reached on the (approved) compute path.
    from experiments.dataset_sensitivity.similarity_ladder import _rotate, _gauss_blur
    out = []
    for i in range(images.shape[0]):
        img = images[i]
        if kind == "rot":
            out.append(_rotate(img, deg=param))
        elif kind == "blur":
            out.append(_gauss_blur(img, sigma=param))
        else:
            raise ValueError(f"unknown shared-transform kind {kind!r}")
    return torch.stack(out, dim=0)


# --------------------------------------------------------------------------- #
# RECOVERY-ERROR METRIC (yoado-6d spec — defined; do NOT redefine downstream). #
# --------------------------------------------------------------------------- #
def recovery_error(kind, theta_hat, theta_true):  # yoado-6d defines this
    """Absolute PARAMETER error in native units for ONE recovery.

    yoado-6d spec:
      * rotation (kind='rot'): CIRCULAR error = min(|theta_hat-theta| mod 360,
        360 - that). Degrees.
      * blur (kind='blur'): plain |sigma_hat - sigma|. Sigma units.
    """
    if kind == "rot":
        d = abs(theta_hat - theta_true) % 360.0
        return min(d, 360.0 - d)
    if kind == "blur":
        return abs(theta_hat - theta_true)
    raise ValueError(f"unknown kind {kind!r}")


def blind_baseline_error(kind, theta_true_draws, prior_mean):
    """Transform-BLIND guesser: always guess the prior MEAN of the parameter.
    Its expected error ~ the prior's mean-abs-deviation (~ prior std over the
    tested range). Returns the list of per-draw baseline errors so the CI is
    over the SAME independent theta-draws as the recovery errors."""
    return [recovery_error(kind, prior_mean, tt) for tt in theta_true_draws]


def recovery_skill(err, err_baseline):
    """Normalized dimensionless headline: skill = 1 - err/err_baseline.
    1 = perfect, 0 = no better than the transform-blind guess, <0 = worse.
    Meaningful recovery REQUIRES err significantly below err_baseline (skill>0)."""
    if err_baseline == 0:
        return float("nan")
    return 1.0 - err / err_baseline


def bootstrap_skill_ci(errs, baseline_errs, n_boot=10000, alpha=0.05, seed=0):
    """Bootstrap 95% CI for the recovery SKILL.

    GUARD (yoado-6d): resample over INDEPENDENT theta-DRAWS (each draw = a fresh
    shared-theta dataset + one recovery), n = number of draws. Do NOT bootstrap
    over the N images — they SHARE theta, so a CI over N is pseudo-replication.
    `errs` / `baseline_errs` must therefore each hold ONE value per theta-draw.
    """
    import random
    assert len(errs) == len(baseline_errs), "errs and baseline_errs must align by draw"
    m = len(errs)
    if m == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    point = recovery_skill(sum(errs) / m, sum(baseline_errs) / m)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(m) for _ in range(m)]  # resample DRAWS, not images
        e = sum(errs[i] for i in idx) / m
        b = sum(baseline_errs[i] for i in idx) / m
        boots.append(recovery_skill(e, b))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return (point, lo, hi)


# --------------------------------------------------------------------------- #
# COMPUTE PATH — SCAFFOLD ONLY. Wired by the executer AFTER Gal approves.       #
# --------------------------------------------------------------------------- #
def run_shared_perturbation_experiment(kind, strengths, n_draws, device, out_path):
    """WOULD (once approved): for each transform strength and each independent
    theta-draw — apply the shared transform to all N images, fine-tune the LoRA
    adapter, run the attack to recover theta_hat, score recovery_error + skill vs
    the transform-blind baseline, bootstrap the skill CI over draws, then plot
    skill (with CI) vs transform strength.

    Intentionally a STUB: even under --approved this raises rather than silently
    launching heavy compute. The executer replaces this body with the fine-tune +
    attack wiring (reuse arm_b_dilution.train_adapter + the whitened attack) after
    Gal signs off on compute.
    """
    raise NotImplementedError(
        "F5 compute path is scaffold-only. The fine-tune + attack + recovery "
        "pipeline is not wired yet; the executer implements it after Gal approves "
        "compute. Metric functions (recovery_error / blind_baseline_error / "
        "recovery_skill / bootstrap_skill_ci) and apply_shared_transform ARE ready."
    )


def _gate_and_maybe_run(args):
    """Refuse the compute path unless --approved is explicitly passed."""
    if not args.approved:
        print("COMPUTE-GATED: awaiting Gal approval; scaffold only", flush=True)
        return 0
    # Even when approved, the pipeline is an explicit NotImplementedError stub so
    # nothing heavy fires by accident from this scaffold.
    run_shared_perturbation_experiment(
        kind=args.kind,
        strengths=args.strengths,
        n_draws=args.n_draws,
        device=args.device,
        out_path=args.out,
    )
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="F5 shared-perturbation recovery (SCAFFOLD ONLY, compute-gated).")
    ap.add_argument("--approved", action="store_true",
                    help="REQUIRED to reach the compute path. Without it the script "
                         "prints the gate message and exits 0 (fires nothing).")
    ap.add_argument("--kind", choices=["rot", "blur"], default="rot",
                    help="shared transform: rot (degrees) or blur (sigma).")
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[2.0, 5.0, 10.0, 15.0, 30.0],
                    help="transform strengths to sweep (deg for rot, sigma for blur).")
    ap.add_argument("--n_draws", type=int, default=20,
                    help="INDEPENDENT theta-draws per strength (bootstrap unit).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(FIGURES, "f5_shared_perturbation.png"))
    args = ap.parse_args()

    # suptitle/caption placeholder carried with the scaffold (rendered once wired).
    print("[F5] figure caption placeholder:", flush=True)
    print("     " + CAPTION, flush=True)
    return _gate_and_maybe_run(args)


if __name__ == "__main__":
    sys.exit(main())
