#!/usr/bin/env python3
"""sigma_i = ||B_T a~_i|| split into the OWN imprint term and the CROSS term from other images' A-drift (audit 2026-09-06):
B_T = sum_j C_j, so B_T a~_i = C_i a~_i + sum_{j != i} C_j a~_i; with frozen A the cross term is exactly 0 (a~_i is
orthogonal to A0 h_j), so its size measures how far sigma_i is from the pure record strength. Also ||a_i^perp|| / ||a_i||.
  python -m experiments.record_strength.sigma_decomposition
"""
import glob, json, os, torch
torch.set_default_dtype(torch.float64)
rows = []
for f in sorted(glob.glob("results/record_strength/*fp64*.pth")):
    d = torch.load(f, weights_only=False, map_location="cpu"); A0, H, B_T, Cimp = d["A0"], d["H"], d["B_T"], d["imprint_C"]; N = H.shape[1]
    a = A0 @ H
    for i in range(N):
        others = torch.cat([a[:, :i], a[:, i + 1:]], 1); Q, _ = torch.linalg.qr(others)
        at = a[:, i] - Q @ (Q.T @ a[:, i]); perp_frac = float(at.norm() / a[:, i].norm()); at = at / at.norm()
        own = float((Cimp[i] @ at).norm()); cross = float(((Cimp.sum(0) - Cimp[i]) @ at).norm()); tot = float((B_T @ at).norm())
        rows.append(dict(file=os.path.basename(f), cell=d["meta"]["cell"], k=d["meta"]["k"], i=i, sigma_i=tot, own=own, cross=cross, cross_over_sigma=cross / tot if tot > 0 else None, perp_frac=perp_frac))
json.dump(rows, open("results/record_strength/sigma_decomposition.json", "w"), indent=1)
print("| cell | k | i | sigma_i | own term | cross term | cross/sigma | ‖a_i^⊥‖/‖a_i‖ |\n|---|---|---|---|---|---|---|---|")
for r in rows: print(f"| {r['cell']} | {r['k']} | {r['i']} | {r['sigma_i']:.2e} | {r['own']:.2e} | {r['cross']:.2e} | {r['cross_over_sigma']:.2e} | {r['perp_frac']:.3f} |")
