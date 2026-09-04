"""Do the per-layer certificates COMPOSE into constraints on the input? (Yoad/yoado-cd, §11.)

One layer's certificate gives `r_l - N'_l` conditions on that layer's own inputs -- about 8 in the standard cell,
against replay's `(m-1) + r - N'` ~ 17. **An image is not 8 numbers**, so a single-layer certificate is not an
image attack and never was. The only route to an image-level budget from the recipe-free side is MANY layers: each
adapted layer constrains its own `h^l`, every `h^l` is a function of the candidate `x` through the released model,
so in principle the conditions add. Whether they add in practice is the question -- the composition runs through
the drift at each depth and through the nonlinearities, and per-layer conditions could be redundant rather than
additive.

Design: an adapted MLP with several adapted layers, rank generous enough that every margin `r_l - N'_l` is
comfortable, and a recovery whose objective is the SUM of the per-layer certificate residuals rather than the first
layer's alone. Layers are added to the objective one at a time.

Pre-registered (before any row):
  * ADDITIVE: recovery error falls as layers are added, and falls below what the single-layer budget allows --
    the operational meaning of "more equations". Report the budget actually used, `sum_l (r_l - N'_l)` over layers
    with a comfortable margin, beside the error.
  * REDUNDANT: the error does not fall. That kills the multi-layer route and is worth knowing fast.
  * A layer whose margin is <= 0 contributes NOTHING (its certificate is the zero matrix) and is excluded from the
    budget count -- the vacuity flag from the per-layer run, applied here as a precondition rather than discovered.
Every row carries the start model and its claim class: these are RANDOM public-scale starts, so the rows are attack
results rather than identifiability ones.
"""
import argparse, json, math, socket, sys, time
import torch, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import PCAChart, read_idx
from experiments.exact_inversion.certificate import certificate, lm_cert
from experiments.exact_inversion.multilayer_lora import forward_adapted, run_training, GELU

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--starts", type=int, default=200); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    ck = torch.load(a.model, map_location=dev, weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    Ws = [sd[k].to(dev).double() for k in sorted(x for x in sd if x.endswith("weight"))]
    b1 = sd[[k for k in sd if k.endswith("bias")][0]].to(dev).double()
    m = Ws[-1].shape[0]
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(Ws[0].shape[1])
    g = torch.Generator().manual_seed(a.seed + 7)
    idx = torch.randperm(Xte_t.shape[0], generator=g)[:a.N].to(dev)
    chart = PCAChart(Xtr_t, a.k, dev)
    X_real = Xte_t[idx].T.contiguous(); X_on = chart.psi(chart.coords_of(X_real)); y = yte_t[idx]
    coord_std = chart.coords_of(Xtr_t[:10000].T).std(dim=1, keepdim=True)
    print(f"# multilayer_budget m={m} N={a.N} r={a.r} k={a.k} T={a.T} lr={a.lr} git={git_hash()} dev={dev}", flush=True)

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    gA = torch.Generator().manual_seed(a.seed + 11)
    A0s = [(a.sigma0 * torch.randn(a.r, W.shape[1], generator=gA)).to(dev) for W in Ws]
    As, Bs = run_training(X_on, Ws, b1, A0s, y, m, a.T, a.lr)

    def inputs_of(x):                                                  # this candidate's inputs at every layer
        h0 = x
        h1 = GELU(Ws[0] @ h0 + b1[:, None] + Bs[0] @ (As[0] @ h0))
        h2 = GELU(Ws[1] @ h1 + Bs[1] @ (As[1] @ h1))
        return [h0, h1, h2]

    Cs, margins, Nps = [], [], []
    H_true = inputs_of(X_on)
    for l in range(3):
        C, Np, _ = certificate(As[l], Bs[l]); Cs.append(C); Nps.append(Np); margins.append(a.r - Np)
        res = torch.linalg.norm(C @ H_true[l], dim=0) / torch.linalg.norm(As[l] @ H_true[l], dim=0)
        rec = [i for i in range(a.N)]
        emit(dict(part="LAYER", layer=l + 1, r=a.r, n_prime=Np, certificate_margin=a.r - Np,
                  usable=bool(a.r - Np > 0), cert_residual_median=float(res.median()),
                  cert_residual_max=float(res.max()), git=git_hash()))
        print(f"  layer {l+1}: N'={Np} margin={a.r-Np} cert residual median {float(res.median()):.2e}", flush=True)
    usable = [l for l in range(3) if margins[l] > 0]
    budget = sum(margins[l] for l in usable)
    print(f"  usable layers {[l+1 for l in usable]}  budget sum(r - N') = {budget} conditions per image "
          f"(replay's per-image budget for comparison: {(m-1) + a.r - Nps[-1]})", flush=True)

    # ---- PRIMARY: rank of the stacked per-layer Jacobian at the truth, layer by layer
    for L in range(1, len(usable) + 1):
        layers = usable[:L]
        def g_stack(w):
            hs = inputs_of(chart.psi(w.reshape(a.k, 1)))
            return torch.cat([(Cs[l] @ hs[l]).reshape(-1) / torch.linalg.norm(As[l] @ hs[l]) for l in layers])
        ranks = []
        for i in range(a.N):                                            # per recorded image, at ITS truth
            w_i = chart.coords_of(X_on[:, i:i + 1]).reshape(-1)
            J = tf.jacfwd(g_stack)(w_i).detach()
            sv = torch.linalg.svdvals(J)
            rk = int((sv > 1e-10 * sv[0]).sum()) if float(sv[0]) > 0 else 0
            ranks.append(dict(image=i, rank=rk, sigma_max=float(sv[0]), sigma_min_nonzero=float(sv[rk - 1]) if rk else 0.0,
                              sigma_rel=[float(v / sv[0]) for v in sv[:min(12, len(sv))]] if float(sv[0]) > 0 else []))
        rk_med = sorted(x["rank"] for x in ranks)[len(ranks) // 2]
        budget_L = sum(margins[l] for l in layers)
        emit(dict(part="RANK", layers_in_objective=[l + 1 for l in layers], n_layers=L, k=a.k, r=a.r, N=a.N,
                  budget_sum_margins=budget_L, ceiling_k=a.k, independent_conditions_median=rk_med,
                  per_image=ranks, saturated_below_sum=bool(rk_med < budget_L), saturated_at_k=bool(rk_med >= a.k),
                  generalised_line_holds=bool(a.k < budget_L),
                  note="independent conditions on the candidate cannot exceed k; the honest budget is min(sum margins, k)",
                  start_model="n/a (Jacobian at the truth, no solve)", claim_class="algebraic check", git=git_hash()))
        print(f"  RANK over layers {[l+1 for l in layers]}: {rk_med} independent conditions "
              f"(sum of margins {budget_L}, ceiling k={a.k})", flush=True)

    # ---- recovery with the objective built from the first `L` usable layers, L = 1, 2, ...
    for L in range(1, len(usable) + 1):
        layers = usable[:L]
        def obj(w):
            hs = inputs_of(chart.psi(w.reshape(a.k, 1)))
            parts = [(Cs[l] @ hs[l]).reshape(-1) / torch.linalg.norm(As[l] @ hs[l]) for l in layers]
            return torch.cat(parts)
        gs = torch.Generator().manual_seed(a.seed + 31); errs = []; t0 = time.time()
        for s_i in range(a.starts):
            w0 = (torch.randn(a.k, 1, generator=gs).to(dev) * coord_std).reshape(-1)
            w, o, _ = lm_cert(obj, w0, a.iters)
            xh = chart.psi(w.reshape(a.k, 1))[:, 0]
            e = min(float(torch.linalg.norm(xh - X_on[:, i]) / torch.linalg.norm(X_on[:, i])) for i in range(a.N))
            errs.append(e)
        se = sorted(errs)
        emit(dict(part="RECOVERY", layers_in_objective=[l + 1 for l in layers], n_layers=L,
                  budget_conditions=sum(margins[l] for l in layers), r=a.r, k=a.k, N=a.N,
                  starts=a.starts, err_min=se[0], err_p10=se[len(se) // 10], err_median=se[len(se) // 2],
                  frac_landed_1e2=float(sum(1 for e in errs if e < 1e-2) / len(errs)),
                  start_model="random public-scale (attacker-buildable)", claim_class="attack",
                  seconds=time.time() - t0, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))
        print(f"  objective over layers {[l+1 for l in layers]} ({sum(margins[l] for l in layers)} conditions): "
              f"median err {se[len(se)//2]:.3e}, best {se[0]:.2e}, landed {sum(1 for e in errs if e < 1e-2)}/{a.starts}", flush=True)


if __name__ == "__main__":
    main()
