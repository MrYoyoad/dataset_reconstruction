#!/usr/bin/env python3
"""LoRA on ALL THREE layers of a trained MLP, inverted by full unrolled backprop.  EMPIRICAL, outside the theorems.

The user's setting: "a trained MLP that got a LoRA update with new samples, then check whether it works."
The head-only cell (trained_backbone.py) is the theorem's setting.  This is the other reading, and it is
deliberately OUTSIDE the theory:

  (A1) FAILS.  Once the first layer is adapted the features reaching the later layers MOVE during training,
  so the closure of Thm 1 does not hold and `simulate_sgd_reduced` is invalid here.  The seed no longer
  enters through an r x N projection, and the released factors are not a function of (data, X).  Everything
  below is therefore empirical, and no capacity line is claimed.

  Consequence for the attack: the nuisance is the FULL A0 of every adapted layer.
      784->1000 : 8 x 784  = 6272        1000->1000 : 8 x 1000 = 8000
      1000->10  : 8 x 1000 = 8000        total 22272, against 112 data unknowns at N=8, k=14.
  An LM Jacobian with 22384 columns is not feasible (the head-only cell had 224), so:

    ARM A  seeds KNOWN  -> unknowns = Nk = 112, solved by LM.  Isolates ONE question: does the multi-layer
                          unrolled inversion recover the data at all when three layers are adapted?
                          This is an ORACLE arm; it is not an attack.
    ARM B  seeds UNKNOWN -> unknowns = 22384, solved by LBFGS.  The honest attack, and the one whose
                          failure mode (search vs alias) is the result if it fails.

Backbone, chart and data are exactly trained_backbone.py's: frozen trained 784-1000-1000-10 GELU MNIST MLP,
private digits from the TEST split (unseen), public PCA chart from the train split.
"""
import argparse, json, math, os, socket, sys, time
import torch, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import git_hash, RECOVER_TOL
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx

torch.set_default_dtype(torch.float64)
GELU = torch.nn.functional.gelu


def forward_adapted(x, Ws, b1, As, Bs):
    """Frozen MLP with a LoRA branch on every layer.  x: (784, N) -> logits (10, N)."""
    h = Ws[0] @ x + b1[:, None] + Bs[0] @ (As[0] @ x)
    h = GELU(h)
    h = Ws[1] @ h + Bs[1] @ (As[1] @ h)
    h = GELU(h)
    return Ws[2] @ h + Bs[2] @ (As[2] @ h)


def run_training(x, Ws, b1, A0s, y, m, T, lr, create_graph=False):
    """Plain SGD on ALL LoRA parameters, unrolled.  B starts at zero on every layer."""
    # autograd.grad needs every LoRA factor to require grad.  Job 608693 died here ("element 0 of tensors does
    # not require grad"): the release was generated under torch.no_grad() with plain A0 leaves and zero Bs.
    # With create_graph the factors stay in the graph (their dependence on x / A0 is what the inversion
    # differentiates through); without it each step re-leafs them so the graph does not grow with T.
    As = [a if a.requires_grad else a.detach().requires_grad_(True) for a in A0s]
    Bs = [torch.zeros(w.shape[0], a.shape[0], dtype=x.dtype, device=x.device, requires_grad=True)
          for w, a in zip(Ws, A0s)]
    Y = torch.eye(m, device=x.device)[y].T
    N = x.shape[1]
    with torch.enable_grad():
        for _ in range(T):
            z = forward_adapted(x, Ws, b1, As, Bs)
            zs = z - z.max(dim=0, keepdim=True).values
            p = torch.exp(zs); p = p / p.sum(dim=0, keepdim=True)
            loss = -(Y * torch.log(p + 1e-300)).sum() / N
            gs = torch.autograd.grad(loss, As + Bs, create_graph=create_graph)
            nA = len(As)
            As = [a - lr * g for a, g in zip(As, gs[:nA])]
            Bs = [b - lr * g for b, g in zip(Bs, gs[nA:])]
            if not create_graph:
                As = [a.detach().requires_grad_(True) for a in As]
                Bs = [b.detach().requires_grad_(True) for b in Bs]
    if not create_graph:
        As = [a.detach() for a in As]; Bs = [b.detach() for b in Bs]
    return As, Bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dataset_reconstruction/models/weights-mnist10_gelu.pth")
    ap.add_argument("--k", type=int, default=14); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--T", type=int, default=100); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--arms", nargs="*", default=["A", "B"])
    ap.add_argument("--lm-iters", type=int, default=60)
    ap.add_argument("--lbfgs-outer", type=int, default=30); ap.add_argument("--lbfgs-iter", type=int, default=40)
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    bb = TrainedBackbone(a.model, dev, "gelu")
    m = bb.m
    Ws = [bb.W1, bb.W2, bb.W0]; b1 = bb.b1
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(1000)

    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev)
    yte_t = torch.tensor(yte, device=dev)
    with torch.no_grad():
        acc = float((bb.logits(Xte_t[:2000].T).argmax(0) == yte_t[:2000]).double().mean())
    g = torch.Generator().manual_seed(a.seed + 7)
    idx = torch.randperm(Xte_t.shape[0], generator=g)[:a.N].to(dev)
    X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
    chart = PCAChart(Xtr_t, a.k, dev)
    W_true = chart.coords_of(X_real); X_img = chart.psi(W_true)
    chart_err = float((torch.linalg.norm(X_img - X_real, dim=0) / torch.linalg.norm(X_real, dim=0)).median())

    dims = [(w.shape[1], w.shape[0]) for w in Ws]
    A0s = [(a.sigma0 * torch.randn(a.r, i, generator=g)).to(dev) for i, o in dims]
    A_T, B_T = run_training(X_img, Ws, b1, A0s, y, m, a.T, a.lr)
    n_seed = sum(a.r * i for i, o in dims); n_data = a.N * a.k
    n_rel = sum(o * a.r for i, o in dims) + n_seed
    print(f"# MULTI-LAYER LoRA on the TRAINED MLP -- EMPIRICAL, outside the theorems ((A1) fails: adapting "
          f"the first layer moves the features)", flush=True)
    print(f"# backbone test acc {acc*100:.2f}%   r={a.r} T={a.T} N={a.N} k={a.k}", flush=True)
    print(f"# unknowns: data {n_data} + seeds {n_seed} = {n_data+n_seed};  released numbers {n_rel}", flush=True)

    def rel_err(Ah, Bh):
        num = sum(float(torch.linalg.norm(x - y_) ** 2) for x, y_ in zip(Ah + Bh, A_T + B_T))
        den = sum(float(torch.linalg.norm(y_) ** 2) for y_ in A_T + B_T)
        return math.sqrt(num / den)

    with torch.no_grad():
        Ah, Bh = run_training(X_img, Ws, b1, A0s, y, m, a.T, a.lr)
    gate = rel_err(Ah, Bh)
    print(f"# GATE  simulator reproduces the release at the truth: {gate:.3e}  "
          f"{'PASS' if gate < 1e-12 else 'FAIL'}", flush=True)
    if gate >= 1e-12: return

    W_init = W_true + a.init_noise * torch.randn(a.k, a.N, generator=g).to(dev) * W_true.std()

    def loss_of(Wc, A0c):
        Ah, Bh = run_training(chart.psi(Wc), Ws, b1, A0c, y, m, a.T, a.lr, create_graph=True)
        num = sum(((x - y_) ** 2).sum() for x, y_ in zip(Ah + Bh, A_T + B_T))
        den = sum(float((y_ ** 2).sum()) for y_ in A_T + B_T)
        return num / den

    for arm in a.arms:
        t0 = time.time()
        if arm == "A":                       # seeds KNOWN -- an ORACLE arm, not an attack
            Wv = W_init.clone().requires_grad_(True)
            opt = torch.optim.LBFGS([Wv], lr=1.0, max_iter=a.lbfgs_iter, history_size=50,
                                    line_search_fn="strong_wolfe")
            for _ in range(a.lbfgs_outer):
                def cl():
                    opt.zero_grad(); f = loss_of(Wv, A0s); f.backward(); return f
                opt.step(cl)
                with torch.no_grad():
                    if float(loss_of(Wv, A0s)) < 1e-28: break
            Wh = Wv.detach(); f = float(loss_of(Wh, A0s)); unk = n_data
        else:                                # seeds UNKNOWN -- the honest attack
            Wv = W_init.clone().requires_grad_(True)
            A0v = [(A0s[j] + 0.0 * torch.randn_like(A0s[j])).clone().requires_grad_(True) if False
                   else (a.sigma0 * torch.randn(a.r, dims[j][0], generator=g)).to(dev).requires_grad_(True)
                   for j in range(3)]
            opt = torch.optim.LBFGS([Wv] + A0v, lr=1.0, max_iter=a.lbfgs_iter, history_size=50,
                                    line_search_fn="strong_wolfe")
            for _ in range(a.lbfgs_outer):
                def cl():
                    opt.zero_grad(); f = loss_of(Wv, A0v); f.backward(); return f
                opt.step(cl)
                with torch.no_grad():
                    if float(loss_of(Wv, A0v)) < 1e-28: break
            Wh = Wv.detach(); f = float(loss_of(Wh, [x.detach() for x in A0v])); unk = n_data + n_seed
        Xh = chart.psi(Wh)
        e_chart = (torch.linalg.norm(Xh - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))
        e_real = (torch.linalg.norm(Xh - X_real, dim=0) / torch.linalg.norm(X_real, dim=0))
        out = dict(setting="multilayer_lora_trained_backbone", arm=arm,
                   arm_desc=("seeds KNOWN (oracle, isolates the unrolled inversion)" if arm == "A"
                             else "seeds UNKNOWN (the honest attack)"),
                   theorems_apply=False, why="(A1) fails: adapting layer 1 moves the features",
                   backbone_test_acc=acc, k=a.k, N=a.N, r=a.r, m=m, T=a.T, lr=a.lr, seed=a.seed,
                   unknowns=unk, unknowns_data=n_data, unknowns_seed=n_seed, released_numbers=n_rel,
                   gate_fwd_check=gate, chart_repr_err=chart_err, residual=f,
                   err_vs_chart_max=float(e_chart.max()), err_vs_REAL_max=float(e_real.max()),
                   err_vs_REAL_median=float(e_real.median()),
                   frac_recovered=float((e_chart < RECOVER_TOL).double().mean()),
                   recovered=bool(float(e_chart.max()) < RECOVER_TOL),
                   seconds=time.time() - t0, git=git_hash(), host=socket.gethostname(),
                   cmd=" ".join(sys.argv))
        print(json.dumps(out), flush=True)
        if a.out:
            with open(a.out, "a") as fh: fh.write(json.dumps(out) + "\n")
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            torch.save(dict(x_real=X_real.cpu(), x_chart=X_img.cpu(), x_hat=Xh.cpu(), meta=out),
                       os.path.join(a.save_dir, f"multilayer_{arm}_k{a.k}_r{a.r}_N{a.N}_s{a.seed}.pth"))


if __name__ == "__main__":
    main()
