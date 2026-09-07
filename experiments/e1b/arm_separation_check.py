#!/usr/bin/env python3
"""Is the seed_free arm actually moving A_0, or is it a relabelled copy of seed_known?

Both arms reported an IDENTICAL median objective to three significant figures at their first checkpoint. The
benign reading is that both fail the same way and the residual is dominated by the H error, fourteen orders above
the truth, so the seed difference is invisible in three digits. The malignant reading is that the seed is not
moving at all. This decides it in two minutes instead of four hours, on the SAME starts the real arms use.

Reports, per start and per iteration: the objective in each arm, and how far A_0 has travelled in the free arm.

  python -m experiments.e1b.arm_separation_check
"""
import torch
from experiments.e1b.e1b_tiny import build_release, replay

torch.set_default_dtype(torch.float64)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k, N, r, m, P, T, lr, seed = 12, 8, 24, 20, 64, 400, 0.05, 1
    R = build_release(k, N, r, m, P, T, lr, seed, dev)
    H, A0, W0, y, A_T, B_T = R["H"], R["A0"], R["W0"], R["y"], R["A_T"], R["B_T"]
    d, N = H.shape
    nB, nAH = torch.linalg.norm(B_T), torch.linalg.norm(A_T @ H)
    nA0 = float(torch.linalg.norm(A0))

    def resid(Hc, A0c):
        A_s, B_s = replay(Hc, A0c, W0, y, m, T, lr)
        return torch.cat([((B_s - B_T) / nB).reshape(-1), ((A_s @ Hc - A_T @ H) / nAH).reshape(-1)])

    print(f"# ||A_0|| = {nA0:.4f}   ||A_T - A_0|| / ||A_0|| = {float(torch.linalg.norm(A_T - A0) / nA0):.4e}")
    print("# the free arm STARTS at A_T, so its seed error begins at that value and should MOVE\n")

    gs = torch.Generator().manual_seed(seed + 31)          # the SAME starts the real arms use
    scale = float(H.norm(dim=0).median())
    G0 = torch.randn(60, d, N, generator=gs).to(dev)
    G0 = G0 / G0.norm(dim=1, keepdim=True) * scale

    for s in range(3):
        print(f"--- start {s} ---")
        traces = {}
        for arm in ("seed_known", "seed_free"):
            Hc = G0[s].clone().requires_grad_(True); params = [Hc]
            A0c = A_T.clone().requires_grad_(True) if arm == "seed_free" else A0
            if arm == "seed_free": params.append(A0c)
            opt = torch.optim.Adam(params, 5e-2)
            H_start = Hc.detach().clone(); A_start = A0c.detach().clone()
            obj = []; g0 = None
            for it in range(30):
                f = resid(Hc, A0c); loss = f @ f
                opt.zero_grad(); loss.backward()
                if it == 0:   # per-block gradient norms, RELATIVE to each block's own scale
                    gh = float(torch.linalg.norm(Hc.grad)) / float(torch.linalg.norm(Hc.detach()))
                    ga = (float(torch.linalg.norm(A0c.grad)) / float(torch.linalg.norm(A0c.detach()))
                          if A0c.grad is not None else 0.0)
                    g0 = (gh, ga)
                opt.step()
                if it % 10 == 9:
                    with torch.no_grad():
                        obj.append((it + 1, float(torch.linalg.norm(resid(Hc, A0c))),
                                    float(torch.linalg.norm(A0c - A0) / nA0),
                                    float(torch.linalg.norm(Hc - H_start) / torch.linalg.norm(H_start)),
                                    float(torch.linalg.norm(A0c - A_start) / torch.linalg.norm(A_start))))
            traces[arm] = obj
            print(f"   {arm:11s} step-0 RELATIVE gradient norms:  H block {g0[0]:.3e}   seed block {g0[1]:.3e}"
                  + ("   <- seed block has NO gradient" if g0[1] == 0.0 else ""))
            for it, o, se, dh, da in obj:
                print(f"   {arm:11s} it={it:<3d} objective {o:.10e}   seed error {se:.4e}   "
                      f"moved: H {dh:.3e}, seed {da:.3e}")
        a, b = traces["seed_known"], traces["seed_free"]
        same = all(abs(x[1] - y[1]) < 1e-15 * max(abs(x[1]), 1e-300) for x, y in zip(a, b))
        moved = b[-1][4] > 0.0
        print(f"   => objectives bitwise-identical across arms: {same}   |   free arm's seed MOVED: {moved}")
        print(f"   => relative objective difference at it=30: "
              f"{abs(a[-1][1] - b[-1][1]) / max(a[-1][1], 1e-300):.3e}\n")
    print("# HOW TO READ THIS. Three outcomes, three different remedies:")
    print("#   (a) seed block has NO gradient, or moved = 0            -> BUG: the arm is a relabelled copy.")
    print("#   (b) seed moves comparably to H (same order relative)    -> arms genuinely separated; identical")
    print("#       medians then mean both fail the same way, with the residual dominated by the H error.")
    print("#   (c) seed moves by orders LESS than H, relative to each  -> UNDER-CONDITIONED, not broken and not a")
    print("#       copy. Remedy is per-block learning rates, not a bug hunt. (Mirror of the VarPro zero-gradient")
    print("#       handicap: there the latent block was starved at step 0; here the seed block would be.)")
    print("# The bitwise test can CONVICT (identical => bug) but cannot ACQUIT: a seed moving by 1e-12 also")
    print("# perturbs later digits while doing no work. The direct measurement is the seed-movement column.")


if __name__ == "__main__":
    main()
