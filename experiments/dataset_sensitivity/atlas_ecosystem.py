"""Ecosystem-attack PROTOTYPE (user idea 2026-08-30): many LoRAs on a shared base θ0 — does removing the
POPULATION's shared common-mode subspace ISOLATE/amplify the private per-adapter signal?

Mechanism (PerPCA / common-mode rejection): each ΔW = shared-base-geometry (common across the population)
+ private residual. If most energy is shared and the private signal is ORTHOGONAL to it, subtracting the
common mode boosts the private SNR by 1/(1-shared_fraction). Test on our factorial zoo (a population on one
base): (a) shared-energy fraction of the top-p population PCs; (b) does composition-recovery SURVIVE the
subtraction (ARI on residual vs raw) — survival ⇒ private ⊥ common-mode ⇒ the ecosystem premise holds.
OBSERVE-framed, weakest→stronger-attacker (population access). Caveat: our zoo's composition signal is
already saturated (distinct digit-sets), so this tests the PREMISE (survival + shared-fraction), not a
recovery GAIN — the gain test needs a weak-signal substrate.
"""
import numpy as np
from experiments.dataset_sensitivity.atlas_analyze import _load, dw_distance, assoc


def main():
    bank, meta = _load("results/atlas_zoo/zoo_bank.pth")
    n = len(bank)
    X = np.stack([c["dW"].ravel() for c in bank])          # n × D
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)      # population SVD
    tot = (s ** 2).sum()
    raw = assoc(dw_distance(bank), [c["composition"] for c in bank])
    print(f"[ecosystem] {n} adapters on shared base | baseline composition ARI (no subtraction) = {raw['ARI']:+.3f}")
    print("subtract top-p POPULATION common-mode PCs, then re-test composition on the residual:")
    for pc in [1, 3, 8, 16, 32, 64]:
        pc = min(pc, len(s))
        shared = float((s[:pc] ** 2).sum() / tot)
        Xres = Xc - (U[:, :pc] * s[:pc]) @ Vt[:pc]          # remove projection onto top-pc pop directions
        res_bank = [dict(dW=Xres[i].reshape(bank[i]["dW"].shape), composition=bank[i]["composition"])
                    for i in range(n)]
        r = assoc(dw_distance(res_bank), [c["composition"] for c in res_bank])
        snr_gain = 1.0 / max(1e-6, 1 - shared)
        print(f"  top-{pc:2d}: shared-energy={shared:.3f}  composition-ARI-on-residual={r['ARI']:+.3f} "
              f"p={r['p']:.3f}  (if private⊥common-mode, SNR×{snr_gain:.1f})")


if __name__ == "__main__":
    main()
