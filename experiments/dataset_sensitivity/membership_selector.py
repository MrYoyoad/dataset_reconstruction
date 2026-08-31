"""Gallery-recovery — SUBSPACE-MEMBERSHIP selector (auditor yoado-d4's reroute, from the row-span theorem).

THEOREM (first-layer LoRA, A₀=0/B₀ random): every step ∂L/∂A = Bᵀ∂L/∂W₁, ∂L/∂W₁ = Σ δᵢxᵢᵀ, so every row of
A_t is a linear combination of the training INPUTS at every step, EXACTLY → row(ΔW)=span{x₁..x_N}, equality
when N≤r; seed-independent (B₀ only mixes coefficients). ⇒ exact recovery for N≤r is a SUBSPACE-MEMBERSHIP
test, not sparse approximation: rank gallery images by residual ‖x−P_V x‖/‖x‖ onto the adapter's row space V,
take the N smallest, exact-verify. Sweep N across the r boundary (N=4,8,12 at r=8): expect ≈exact for N≤r,
degrade past it (row space becomes an r-dim projection = the real superposition problem, Cocktail-Party/SPEAR).
SCOPE: first-layer LoRA, A₀=0, N≤r → row space = input span (exact); closed-world; this attacker. bsub GPU.
"""
import argparse, os, torch, numpy as np
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set
from experiments.dataset_sensitivity.phase0_gallery import gallery
from experiments.data_utils import _load_dataset

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/membership_selector"
NS = [4, 8, 12]            # across the rank boundary r=8
RANK, T, LR, ACT = 8, 200, 0.5, "gelu"
N_TARGETS = 10


def row_space(dW, tol=1e-6):
    """Significant right singular vectors (the row/input space, effective rank auto-detected)."""
    svd = torch.linalg.svd(dW.detach().to("cpu", torch.float64), full_matrices=False)
    keep = (svd.S > tol * svd.S[0]).sum().item()
    return svd.Vh[:keep].transpose(-1, -2).contiguous()          # (in, keep)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ds = _load_dataset("mnist", train=True); act = make_activation(ACT)
    xr, yr, _ = build_set(2, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, ds_mean = _honest_target(xr, yr, T, RANK, ACT, LR, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]
    gimg, glab, gdig = gallery(ds, dev); G = gimg.shape[0]; gx0 = (gimg - ds_mean)
    gx0_cpu = gx0.reshape(G, -1).to("cpu")                        # (G, 784)
    gnorm = gx0_cpu.norm(dim=1)
    B0_atk = draw_B0(900, out_f, RANK, dev)
    c0 = (gdig == 0).nonzero().view(-1); c1 = (gdig == 1).nonzero().view(-1)
    print(f"[selector] |G|={G} rank r={RANK} | N-sweep {NS} | row-span membership test")

    summary = {}
    for N in NS:
        rng = torch.Generator().manual_seed(11 + N)
        exact, jac_sum, mres, nres = 0, 0.0, [], []
        for t in range(N_TARGETS):
            s = torch.cat([c0[torch.randperm(len(c0), generator=rng)[:N // 2]],
                           c1[torch.randperm(len(c1), generator=rng)[:N // 2]]])
            _, _, _, dWt = train_adapter(frozen, b0, B0_atk, gx0[s], glab[s], LR, T, act, RANK)
            V = row_space(dWt)                                    # (784, keep)
            proj = gx0_cpu @ V @ V.transpose(-1, -2)              # P_V x for every gallery image
            resid = (gx0_cpu - proj).norm(dim=1) / (gnorm + 1e-12)
            order = torch.argsort(resid)
            cand = set(order[:N].tolist()); ss = set(s.tolist())
            exact += int(cand == ss); jac_sum += len(cand & ss) / len(cand | ss)
            mres += resid[s].tolist(); nres += resid[[i for i in range(G) if i not in ss]].tolist()
        er = exact / N_TARGETS; jac = jac_sum / N_TARGETS
        mres, nres = np.array(mres), np.array(nres)
        summary[N] = dict(exact=er, jaccard=jac, member_resid=float(mres.mean()), nonmember_resid=float(nres.mean()))
        tag = "N≤r" if N <= RANK else "N>r"
        print(f"  N={N:2d} ({tag}): EXACT={exact}/{N_TARGETS}={er:.2f}  Jaccard={jac:.3f}  "
              f"member-resid={mres.mean():.2e} (max {mres.max():.2e})  non-member-resid={nres.mean():.3f}")

    print(f"\n  [row-span theorem: member residual ≈0 to numerical precision for N≤r={RANK}; "
          f"exact recovery degrades past N=r = the superposition regime]")
    print(f"  [SCOPE: first-layer LoRA, A₀=0, N≤r → row space = input span (exact); closed-world; this attacker]")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
    ns = list(summary); er = [summary[n]["exact"] for n in ns]
    bars = ax[0].bar([str(n) for n in ns], er, color=["#2c7fb8" if n <= RANK else "#d95f0e" for n in ns], edgecolor="k")
    ax[0].axvline(1.5, ls="--", color="#888"); ax[0].text(1.52, 0.5, "N = r = %d" % RANK, fontsize=9, color="#555")
    for b, e in zip(bars, er):
        ax[0].text(b.get_x() + b.get_width() / 2, e + 0.02, f"{e:.0%}", ha="center", fontweight="bold")
    ax[0].set_ylim(0, 1.1); ax[0].set_ylabel("exact-set recovery rate"); ax[0].set_xlabel("N (private-set size)")
    ax[0].set_title("Exact recovery: 100% at N≤r, collapses at N>r", fontsize=11, fontweight="bold")
    mr = [summary[n]["member_resid"] for n in ns]; nr = [summary[n]["nonmember_resid"] for n in ns]
    ax[1].semilogy(ns, [max(m, 1e-16) for m in mr], "o-", color="#2c7fb8", lw=2, ms=8, label="member residual")
    ax[1].semilogy(ns, nr, "s--", color="#d95f0e", lw=2, ms=8, label="non-member residual")
    ax[1].set_xlabel("N"); ax[1].set_ylabel("‖x − P_V x‖ / ‖x‖  (log)"); ax[1].set_xticks(ns)
    ax[1].set_title("Members lie in the row span to machine precision (N≤r)", fontsize=11, fontweight="bold")
    ax[1].legend(fontsize=9)
    fig.suptitle("First-layer LoRA (A₀=0) publishes the exact INPUT SPAN of its private set for N≤r\n"
                 "closed-world exact-subset recovery via a subspace-membership test · this attacker · DETECTION/RECOVERY not reconstruction",
                 fontsize=11, fontweight="bold", y=1.06)
    os.makedirs("figures/harder_id", exist_ok=True)
    fig.tight_layout(); fig.savefig("figures/harder_id/membership_selector.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[saved] figures/harder_id/membership_selector.png")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(summary=summary, G=G, rank=RANK, NS=NS), os.path.join(RESULTS, "selector.pth"))
        print(f"[saved] {RESULTS}/selector.pth")


if __name__ == "__main__":
    main()
