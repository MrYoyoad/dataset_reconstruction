"""C — MEMBERSHIP INFERENCE from the adapter (drops the closed-set crutch; literature-comparable).

"Was THIS image in the adapter's private set?" Score s(x)=‖ΔW·(x−μ)‖ (μ = same-distribution {0,1} mean, so the
score isolates the INSTANCE deviation, not "is this a {0,1} image"). Positives = the adapter's N private images;
negatives = the GLOBALLY-HELD-OUT pool (non-members of EVERY adapter — the auditor's fix against pooled-AUC
inflation). MIA AUC per adapter, cluster-robust over private-sets; compared to random≈0.5 and the LoRA-Leak
anchor (0.775, verify at source). DETECTION not reconstruction. numpy/scipy.
"""
import numpy as np, torch, os
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BANK = "results/mia_zoo/mia_bank.pth"


def _auc(labels, scores):
    order = np.argsort(scores); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1; npos = pos.sum(); nneg = (~pos).sum()
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def main():
    d = torch.load(BANK, map_location="cpu", weights_only=False)
    bank = [c for c in d["bank"] if c.get("converged", True)]
    neg = d["neg_pool"].to(torch.float64).numpy(); mu = d["mu"].to(torch.float64).numpy()
    print(f"[C] {len(bank)} adapters | {neg.shape[0]} GLOBAL negatives (non-members of ALL sets) | mu=same-dist {{0,1}} mean")
    aucs, sids = [], []
    for c in bank:
        dW = (c["B"].to(torch.float64).numpy() @ c["A"].to(torch.float64).numpy())
        pos = c["priv_imgs"].to(torch.float64).numpy()
        pool = np.concatenate([pos, neg], axis=0)
        lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        scores = np.linalg.norm((pool - mu) @ dW.T, axis=1)
        aucs.append(_auc(lab, scores)); sids.append(c["set_id"])
    aucs = np.array(aucs); sids = np.array(sids)
    # cluster-robust over private-set
    usid = sorted(set(sids)); cl = np.array([aucs[sids == s].mean() for s in usid])
    G = len(cl); est = float(cl.mean()); se = cl.std(ddof=1) / np.sqrt(G); t = stats.t.ppf(0.975, G - 1)
    ci = (est - t * se, est + t * se)
    print(f"\n=== MEMBERSHIP-INFERENCE AUC (adapter-only, prior-free, same-dist negatives) ===")
    print(f"  MIA AUC = {est:.3f}  CI95 [{ci[0]:.3f}, {ci[1]:.3f}]  (G={G} private-sets)")
    print(f"  random-image floor = 0.500 | LoRA-Leak anchor = 0.775 (VERIFY at source)")
    verdict = ("LEAKS membership (CI excludes 0.5)" if ci[0] > 0.5 else "no membership signal (CI includes 0.5)")
    print(f"  → {verdict}  [DETECTION not reconstruction · weakest-attacker lower bound]")

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
    ax.hist(aucs, bins=np.linspace(0.3, 1.0, 15), color="#2c7fb8", edgecolor="k", alpha=0.85)
    ax.axvline(0.5, color="#888", ls="--", lw=1.5, label="random floor 0.5")
    ax.axvline(0.775, color="#d95f0e", ls=":", lw=1.5, label="LoRA-Leak 0.775")
    ax.axvline(est, color="#0a4d7a", lw=2.5, label=f"mean AUC {est:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")
    ax.set_xlabel("per-adapter membership-inference AUC"); ax.set_ylabel("# adapters")
    ax.set_title("Membership inference from the adapter alone\n"
                 "s(x)=‖ΔW·(x−μ)‖, same-distribution negatives (globally held out)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.text(0.5, -0.24, "DETECTION not reconstruction · adapter-only prior-free = weakest-attacker LOWER bound · "
            "N=4 MNIST-MLP · negatives disjoint from EVERY adapter · cluster-robust over private-sets",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#555")
    os.makedirs("figures/harder_id", exist_ok=True)
    fig.savefig("figures/harder_id/c_membership.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("\n[saved] figures/harder_id/c_membership.png")


if __name__ == "__main__":
    main()
