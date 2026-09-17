"""Independent FP64 evaluation of the three hand-computed coefficients in the archive's multilayer note.

Why this exists
---------------
`notes/gal_2026-09/multilayer_lora_theory_source.tex` states three leading coefficients that were computed by
hand and carry a PROVED status, and none has ever been evaluated numerically in this repository (the existing
`experiments/multilayer_cert/theory_checks.py` T3 check validates a *different* instance). Ruled C4 by the
approver lane at commit e0a2bb1: "a hand-computed constant carrying a PROVED status with no independent
evaluation is the same shape as a numerical check whose harness implements the corrected law -- a status resting
on the author's own arithmetic."

This script re-derives each of the three from the dynamics the note specifies, rather than re-evaluating the
note's own closed forms. Every cell simulates the stated training from its stated initialisation using only the
LoRA gradient rules, then reads the certificate off the simulated release.

    Cell A  Ex 3.5  (tex L333-363)   linear two-layer instance   claim  C e1 = -(3/8) eps e2 + O(eps^2)
    Cell B  Ex 5.5  (tex L733-793)   softplus MLP                claim  C h  = (eta/16) (1,-1)^T + O(eta^2)
    Cell C  Sec 8.6 (tex L1160-1190) two residual blocks         claim  Ctil e1 = -(1/3) eta^2 e2 + O(eta^3)

Each cell also checks the note's *exact* intermediate matrices (which it states in closed form) to machine
precision. Those are the load-bearing half: if the simulated trajectory disagrees with the note's stated
`v_2`, `A_2`, `B_1`, `b_1`, ... then the coefficient is being compared against a different dynamical system and
the coefficient comparison means nothing.

PRE-STATED TOLERANCES (fixed before the first run; do not loosen to make a cell pass)
------------------------------------------------------------------------------------
  EXACT_TOL   = 1e-12   relative, for every closed form the note states exactly
  COEFF_TOL   = 2e-3    relative, for the leading coefficient at the smallest step
  ORDER_TOL   = 0.05    absolute, on the measured log-log slope against the claimed order

A cell PASSES only if all three hold. Reporting rule: a cell that fails the exact check is reported as
`construction mismatch` and its coefficient verdict is withheld -- that is a different outcome from
`coefficient wrong`, and the two must not be merged.

Run (CPU, seconds, no GPU):  python -u -m experiments.archive_checks.check_hand_coefficients
Submit:                      bsub < scripts/run_archive_coeff_check_wexac.sh
Rows land in results/archive_checks/hand_coefficients_<jobid>.jsonl
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np

np.seterr(all="raise")

EXACT_TOL = 1e-12
COEFF_TOL = 2e-3
ORDER_TOL = 0.05

E1 = np.array([1.0, 0.0])
E2 = np.array([0.0, 1.0])


# ----------------------------------------------------------------------------------- helpers


def rel(a, b) -> float:
    """Relative error with a sane denominator (0 when both are 0)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(np.linalg.norm(b), np.linalg.norm(a), 1.0)
    return float(np.linalg.norm(a - b) / denom)


def certificate(B: np.ndarray, A: np.ndarray, truncate_to: int | None = None) -> np.ndarray:
    """C = P_{row(B)^perp} A.

    `truncate_to=k` uses the top-k right-singular subspace of B instead of its full row space --
    the truncated certificate the note uses in Sec 8.6.
    """
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    _, s, vt = np.linalg.svd(B, full_matrices=False)
    if truncate_to is None:
        floor = max(s[0], 1.0) * 1e-12 if s.size else 0.0  # absolute floor, never relative-only
        keep = int(np.sum(s > floor))
    else:
        keep = int(truncate_to)
    if keep == 0:
        return A.copy()
    V = vt[:keep].T
    return A - V @ (V.T @ A)


def leading_coefficient(values: np.ndarray, steps: np.ndarray, order: int):
    """Fit value ~ c * step**order and return (c_at_smallest_step, measured_log_log_slope)."""
    norms = np.linalg.norm(values, axis=1)
    finite = norms > 0
    slope = float(
        np.polyfit(np.log(steps[finite]), np.log(norms[finite]), 1)[0]
    ) if finite.sum() >= 2 else float("nan")
    c = values[-1] / steps[-1] ** order
    return c, slope


def softplus(z):
    return np.log1p(np.exp(z))


def dsoftplus(z):
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------------- cell A


def cell_A(eps: float, eta: float = 0.5):
    """Ex 3.5 (tex L333-363).

    h = e1 + b*a  (upstream rank-one adapter, b in R^2, a scalar)
    f = (e2^T + v A) h   (observed adapter: v in R^{1x2} plays B, A in R^{2x2})
    loss = (f - 1)^2 / 2 at the single input x = 1 with target 1.
    Two SIMULTANEOUS steps: upstream step eps, observed step eta.
    """
    b = np.zeros(2)
    a = 1.0
    v = np.zeros((1, 2))
    A = np.eye(2)

    history = []
    for _ in range(2):
        h = E1 + b * a
        f = float((E2.reshape(1, 2) + v @ A) @ h)
        d = f - 1.0
        # observed layer
        g_v = d * (A @ h).reshape(1, 2)
        g_A = v.T * d @ h.reshape(1, 2)
        # upstream layer
        row = (E2.reshape(1, 2) + v @ A).ravel()
        g_b = d * row * a
        g_a = float(d * row @ b)
        v, A, b, a = v - eta * g_v, A - eta * g_A, b - eps * g_b, a - eps * g_a
        history.append((v.copy(), A.copy(), b.copy(), a))

    (v1, A1, b1, a1), (v2, A2, b2, a2) = history

    # the note's exact statements for this trajectory
    exact = {
        "b_1 = eps*e2": rel(b1, eps * E2),
        "a_1 = 1": rel(a1, 1.0),
        "v_1 = eta*e1^T": rel(v1, eta * E1.reshape(1, 2)),
        "A_1 = I": rel(A1, np.eye(2)),
        "v_2": rel(
            v2,
            eta * (2 - eta - eps) * E1.reshape(1, 2) + eta * eps * (1 - eta - eps) * E2.reshape(1, 2),
        ),
        "A_2": rel(
            A2,
            np.eye(2) + eta**2 * (1 - eta - eps) * np.outer(E1, E1 + eps * E2),
        ),
        "d_1 = eta+eps-1": rel(
            float((E2.reshape(1, 2) + v1 @ A1) @ (E1 + b1 * a1)) - 1.0, eta + eps - 1.0
        ),
    }
    C = certificate(v2, A2)
    return C @ E1, exact


# ----------------------------------------------------------------------------------- cell B


def cell_B(eta: float):
    """Ex 5.5 (tex L733-793).

    z(x) = [(1,0)^T + u*a] x + (-1,0)^T ,  h = softplus(z)      (upstream rank-one: u in R^2, a scalar)
    f(x) = (I + B A) h(x)                                       (observed: B, A in R^{2x2})
    y = h_base - e1 ,  loss = ||f(1) - y||^2 / 2 ,  two SIMULTANEOUS steps of size eta (all factors).
    """
    s = np.log(2.0)
    h_base = s * np.array([1.0, 1.0])
    y = h_base - E1

    u = np.zeros(2)
    a = 1.0
    B = np.zeros((2, 2))
    A = np.eye(2)

    first = None
    for step in range(2):
        z = (E1 + u * a) * 1.0 + (-E1)
        h = softplus(z)
        M = np.eye(2) + B @ A
        D = M @ h - y
        g_B = np.outer(D, A @ h)
        g_A = B.T @ np.outer(D, h)
        dz = (M.T @ D) * dsoftplus(z)
        g_u = dz * a * 1.0
        g_a = float(dz @ (u * 1.0))
        B, A, u, a = B - eta * g_B, A - eta * g_A, u - eta * g_u, a - eta * g_a
        if step == 0:
            first = (B.copy(), A.copy(), u.copy(), a)

    B1, A1, u1, a1 = first
    exact = {
        "u_1 = -(eta/2) e1": rel(u1, -(eta / 2) * E1),
        "a_1 = 1": rel(a1, 1.0),
        "B_1 = -eta e1 h^T": rel(B1, -eta * np.outer(E1, h_base)),
        "A_1 = I": rel(A1, np.eye(2)),
    }
    C = certificate(B, A)
    return C @ h_base, exact


# ----------------------------------------------------------------------------------- cell C


def cell_C(eta: float):
    """Sec 8.6 (tex L1160-1190).

    Two width-two residual blocks on the single input e1, loss ||f||^2/2, common step eta, THREE steps.
    Block 1 adapter: column b in R^2, row a in R^{1x2}, seeded a_0 = e1^T, b_0 = 0.
    Block 2 adapter: B in R^{2x2} (seed 0), A_0 = [[1,1],[0,1]].
    The claim is about the TOP-ONE TRUNCATED certificate of the second block.
    """
    b = np.zeros(2)
    a = E1.reshape(1, 2).copy()
    B = np.zeros((2, 2))
    A0 = np.array([[1.0, 1.0], [0.0, 1.0]])
    A = A0.copy()

    first = None
    for step in range(3):
        h0 = E1.copy()
        h1 = h0 + b @ (a @ h0)
        h2 = h1 + B @ (A @ h1)
        D2 = h2  # dL/df for L = ||f||^2 / 2
        g_B = np.outer(D2, A @ h1)
        g_A = B.T @ np.outer(D2, h1)
        dh1 = D2 + (B @ A).T @ D2
        g_b = dh1 * float(a @ h0)
        g_a = (b @ dh1) * h0.reshape(1, 2)
        B, A, b, a = B - eta * g_B, A - eta * g_A, b - eta * g_b, a - eta * g_a
        if step == 0:
            first = (b.copy(), a.copy(), B.copy(), A.copy())

    b1, a1, B1, A1 = first
    exact = {
        "b_1 = -eta e1": rel(b1, -eta * E1),
        "a_1 = e1^T": rel(a1, E1.reshape(1, 2)),
        "B_1 = -eta e1 e1^T": rel(B1, -eta * np.outer(E1, E1)),
        "A_1 = A_0": rel(A1, A0),
    }
    Ctil = certificate(B, A, truncate_to=1)
    return Ctil @ E1, exact


# ----------------------------------------------------------------------------------- driver

CELLS = [
    dict(
        name="A_ex3.5_linear_first_order",
        tex="notes/gal_2026-09/multilayer_lora_theory_source.tex L333-363",
        order=1,
        claim=-(3.0 / 8.0) * E2,
        claim_text="C e1 = -(3/8) eps e2 + O(eps^2), at eta = 1/2",
        steps=np.array([1e-3, 1e-4, 1e-5, 1e-6]),
        fn=lambda s: cell_A(eps=s, eta=0.5),
    ),
    dict(
        name="B_ex5.5_softplus_first_order",
        tex="notes/gal_2026-09/multilayer_lora_theory_source.tex L733-793",
        order=1,
        claim=(1.0 / 16.0) * np.array([1.0, -1.0]),
        claim_text="C h = (eta/16) (1,-1)^T + O(eta^2)",
        steps=np.array([1e-3, 1e-4, 1e-5, 1e-6]),
        fn=lambda s: cell_B(eta=s),
    ),
    dict(
        name="C_sec8.6_residual_second_order",
        tex="notes/gal_2026-09/multilayer_lora_theory_source.tex L1160-1190",
        order=2,
        claim=-(1.0 / 3.0) * E2,
        claim_text="Ctil_2 e1 = -(1/3) eta^2 e2 + O(eta^3) (top-one truncated)",
        steps=np.array([1e-2, 1e-3, 1e-4, 1e-5]),
        fn=lambda s: cell_C(eta=s),
    ),
]


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    assert np.dtype(float) == np.float64, "FP64 required"
    job = os.environ.get("LSB_JOBID", "local")
    out_dir = os.path.join("results", "archive_checks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"hand_coefficients_{job}.jsonl")

    meta = dict(
        job=job,
        git=git_hash(),
        host=socket.gethostname(),
        python=platform.python_version(),
        numpy=np.__version__,
        cmd=" ".join(sys.argv),
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
        tolerances=dict(exact=EXACT_TOL, coefficient=COEFF_TOL, order=ORDER_TOL),
    )
    failures = 0
    with open(out_path, "w") as fh:
        fh.write(json.dumps(dict(kind="meta", **meta)) + "\n")
        for cell in CELLS:
            values, exacts = [], {}
            for s in cell["steps"]:
                v, ex = cell["fn"](float(s))
                values.append(np.asarray(v, dtype=np.float64))
                exacts = {k: max(exacts.get(k, 0.0), val) for k, val in ex.items()}
            values = np.stack(values)
            worst_exact = max(exacts.values()) if exacts else 0.0
            c, slope = leading_coefficient(values, cell["steps"], cell["order"])
            coeff_err = rel(c, cell["claim"])
            order_err = abs(slope - cell["order"])

            construction_ok = worst_exact <= EXACT_TOL
            if not construction_ok:
                verdict = "construction mismatch (coefficient verdict withheld)"
            elif coeff_err <= COEFF_TOL and order_err <= ORDER_TOL:
                verdict = "confirmed"
            elif order_err > ORDER_TOL:
                verdict = f"wrong order (measured slope {slope:.4f}, claimed {cell['order']})"
            else:
                verdict = "coefficient disagrees"
            if verdict != "confirmed":
                failures += 1

            row = dict(
                kind="cell",
                name=cell["name"],
                tex=cell["tex"],
                claim=cell["claim_text"],
                claimed_vector=cell["claim"].tolist(),
                measured_vector=c.tolist(),
                coefficient_rel_err=coeff_err,
                measured_slope=slope,
                claimed_order=cell["order"],
                worst_exact_rel_err=worst_exact,
                exact_checks={k: v for k, v in sorted(exacts.items())},
                steps=cell["steps"].tolist(),
                verdict=verdict,
            )
            fh.write(json.dumps(row) + "\n")
            print(f"[{cell['name']}] {verdict}")
            print(f"    claim    {cell['claim_text']}")
            print(f"    measured {np.array2string(c, precision=6)}   rel err {coeff_err:.3e}")
            print(f"    slope    {slope:.4f} (claimed {cell['order']})")
            print(f"    exact checks, worst rel err {worst_exact:.3e}:")
            for k, v in sorted(exacts.items()):
                print(f"        {k:<28} {v:.3e}")
        fh.write(json.dumps(dict(kind="summary", cells=len(CELLS), failures=failures)) + "\n")

    print(f"\n{len(CELLS) - failures} of {len(CELLS)} confirmed; rows -> {out_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
