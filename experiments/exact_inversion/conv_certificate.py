#!/usr/bin/env python3
"""Does the recipe-free certificate survive WEIGHT SHARING?  The convolutional comparison.

Everything on the certificate side rests on one count: `rank B_T = N'`, the number of RECORDED items, and the
margin `r - N'` is what is left over to constrain a candidate.  In a dense layer each recorded image contributes
ONE vector `A h_i` to that span, so `N' <= N` and the margin is generous.  A convolution shares its weights across
P spatial positions, so a single image contributes P patch vectors `A h_{ip}` -- and the span they fill is what
decides whether any certificate exists at all.

Written as the whole question:  margin_l = min(r, d_l) - rank{ A h_{ip} : recorded i, all p },  d_l = C_in*k*k.
  * If the recorded patches SPAN their input dimension, `N' = d_l`, `C = 0`, and the certificate is VACUOUS on
    convolutions at every rank -- the route is a dense-layer phenomenon and does not transfer.
  * If they do not (MNIST is mostly identical background, so patch directions are far fewer than N*P), the margin
    is set by the number of DISTINCT patch activations, which is a property of the DATA, not of the image count.
    Then a conv layer supplies margin x P conditions per image, and weight sharing helps rather than hurts.
Both are live; the run decides, and the discriminator is `rank B_T` against `min(r, d_l)` and against `N*P_l`.

PRE-REGISTERED (before any row):
  CONV-VACUOUS   rank B_T = min(r, d_l) at every adapted conv layer for every r tried, so every margin is 0. The
                 pixel-rank curve cannot even be run, and the honest statement becomes: the recipe-free channel is
                 an MLP/token phenomenon; on a downsampling conv path there is no certificate to have.
  CONV-CARRIES   some layer has margin > 0 at some r, in which case the pixel rank of the stacked condition
                 (C_l applied at EVERY spatial position) is measured exactly as for the MLP and put on the same
                 axis -- with the prediction that the curve flattens at the narrowest spatial bottleneck, since
                 every deeper layer's pixel map factors through the downsampled representation.
Rows are algebraic checks at the truth: no solve, no start, not an attack.

  python -u -m experiments.exact_inversion.conv_certificate --ranks 8 16 32 64 128 256 512
"""
import argparse, json, math, os, socket, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import read_idx
from experiments.exact_inversion.certificate import certificate

torch.set_default_dtype(torch.float64)
SPEC = [(1, 32, 3, 2, 1), (32, 64, 3, 2, 1), (64, 64, 3, 2, 1)]     # (cin, cout, k, stride, pad) on 28x28


def patches_of(h, k, stride, pad):
    """(N, C, H, W) -> (N, C*k*k, P), the vectors a shared kernel actually multiplies."""
    return F.unfold(h, k, padding=pad, stride=stride)


def conv_forward(x, Wms, bs, Whead, bhead, As=None, Bs=None, want_patches=False):
    """x: (N, 1, 28, 28).  Wms[l] is (cout, cin*k*k) -- the kernel as the matrix the certificate acts on."""
    h = x; kept = []
    for l, (cin, cout, k, s, p) in enumerate(SPEC):
        P = patches_of(h, k, s, p)
        if want_patches: kept.append(P)
        z = Wms[l] @ P + bs[l][:, None]
        if As is not None: z = z + Bs[l] @ (As[l] @ P)
        side = int(math.isqrt(z.shape[-1]))
        h = F.gelu(z.reshape(z.shape[0], cout, side, side))
    flat = h.reshape(h.shape[0], -1)
    z = flat @ Whead.T + bhead
    if As is not None: z = z + (flat @ As[-1].T) @ Bs[-1].T
    return (z, kept) if want_patches else z


def run_training(x, Wms, bs, Whead, bhead, A0s, y, m, T, lr):
    As = [a.detach().clone().requires_grad_(True) for a in A0s]
    Bs = [torch.zeros(o, a.shape[0], dtype=x.dtype, device=x.device, requires_grad=True)
          for o, a in zip([c for _, c, _, _, _ in SPEC] + [m], A0s)]
    Y = torch.eye(m, device=x.device)[y]
    for _ in range(T):
        z = conv_forward(x, Wms, bs, Whead, bhead, As, Bs)
        zs = z - z.max(dim=1, keepdim=True).values
        p = torch.exp(zs); p = p / p.sum(dim=1, keepdim=True)
        loss = -(Y * torch.log(p + 1e-300)).sum() / x.shape[0]
        gs = torch.autograd.grad(loss, As + Bs)
        n = len(As)
        As = [(a - lr * g).detach().requires_grad_(True) for a, g in zip(As, gs[:n])]
        Bs = [(b - lr * g).detach().requires_grad_(True) for b, g in zip(Bs, gs[n:])]
    return [a.detach() for a in As], [b.detach() for b in Bs]


def train_backbone(Xtr, ytr, Xte, yte, dev, epochs, bs, lr, seed):
    torch.manual_seed(seed)
    convs = nn.ModuleList([nn.Conv2d(ci, co, k, s, p) for ci, co, k, s, p in SPEC]).to(dev).float()
    head = nn.Linear(SPEC[-1][1] * 16, 10).to(dev).float()
    opt = torch.optim.Adam(list(convs.parameters()) + list(head.parameters()), lr=lr)
    lossf = nn.CrossEntropyLoss()
    def f(x):
        h = x
        for c in convs: h = F.gelu(c(h))
        return head(h.reshape(h.shape[0], -1))
    for ep in range(epochs):
        perm = torch.randperm(Xtr.shape[0], device=dev)
        for i in range(0, Xtr.shape[0], bs):
            j = perm[i:i + bs]
            opt.zero_grad(); lossf(f(Xtr[j]), ytr[j]).backward(); opt.step()
        with torch.no_grad():
            acc = float((f(Xte).argmax(1) == yte).double().mean())
        print(f"  epoch {ep+1}  test acc {acc*100:.2f}%", flush=True)
    Wms = [c.weight.detach().double().reshape(c.weight.shape[0], -1) for c in convs]
    bs_ = [c.bias.detach().double() for c in convs]
    return Wms, bs_, head.weight.detach().double(), head.bias.detach().double(), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--ranks", nargs="*", type=int,
                    default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--T", type=int, default=200); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--backbone-lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--pixel-rank", action="store_true", help="only meaningful if some margin is > 0")
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--ckpt", default="models/exact_inversion/mnist_conv.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, ytr = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr, device=dev).float().reshape(-1, 1, 28, 28)
    ytr_t = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev).float().reshape(-1, 1, 28, 28); yte_t = torch.tensor(yte, device=dev)

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    if os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        Wms = [w.to(dev) for w in ck["Wms"]]; bs_ = [b.to(dev) for b in ck["bs"]]
        Whead = ck["Whead"].to(dev); bhead = ck["bhead"].to(dev); acc = ck["test_acc"]
        print(f"# loaded conv backbone {a.ckpt} test acc {acc*100:.2f}%", flush=True)
    else:
        os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
        Wms, bs_, Whead, bhead, acc = train_backbone(Xtr_t, ytr_t, Xte_t, yte_t, dev, a.epochs, a.bs,
                                                     a.backbone_lr, a.seed)
        torch.save(dict(Wms=[w.cpu() for w in Wms], bs=[b.cpu() for b in bs_], Whead=Whead.cpu(),
                        bhead=bhead.cpu(), test_acc=acc, git=git_hash()), a.ckpt)
        print(f"# trained conv backbone -> {a.ckpt} test acc {acc*100:.2f}%", flush=True)
    Wms = [w.double() for w in Wms]; bs_ = [b.double() for b in bs_]
    Whead = Whead.double(); bhead = bhead.double()

    g = torch.Generator().manual_seed(a.seed + 7)
    idx = torch.randperm(Xte_t.shape[0], generator=g)[:a.N].to(dev)
    X = Xte_t[idx].double(); y = yte_t[idx]
    m = 10
    dims = [ci * k * k for ci, co, k, s, p in SPEC] + [SPEC[-1][1] * 16]

    # the count that decides everything, measured on the FROZEN backbone first: how many independent patch
    # directions do N recorded images actually supply at each layer?
    with torch.no_grad():
        _, kept = conv_forward(X, Wms, bs_, Whead, bhead, want_patches=True)
    for l, Pt in enumerate(kept):
        M = Pt.permute(1, 0, 2).reshape(Pt.shape[1], -1)              # (d_l, N*P)
        sv = torch.linalg.svdvals(M)
        rk = int((sv > 1e-10 * sv[0]).sum())
        emit(dict(part="PATCH_SPAN", layer=l + 1, d_l=dims[l], patches_per_image=int(Pt.shape[2]),
                  n_patch_vectors=int(Pt.shape[0] * Pt.shape[2]), patch_span_rank=rk,
                  spans_input_dim=bool(rk >= dims[l]), N=a.N, git=git_hash()))
        print(f"  layer {l+1}: d={dims[l]}  P={Pt.shape[2]}  N*P={Pt.shape[0]*Pt.shape[2]}  "
              f"patch span rank {rk}  {'SPANS d (no margin at any r)' if rk >= dims[l] else 'deficient'}",
              flush=True)

    for r in a.ranks:
        sigma0 = a.sigma0 or 1.0 / math.sqrt(dims[0])
        gA = torch.Generator().manual_seed(a.seed + 11)
        A0s = [(sigma0 * torch.randn(r, d, generator=gA)).to(dev) for d in dims]
        t0 = time.time()
        As, Bs = run_training(X, Wms, bs_, Whead, bhead, A0s, y, m, a.T, a.lr)
        with torch.no_grad():
            _, kept = conv_forward(X, Wms, bs_, Whead, bhead, As, Bs, want_patches=True)
        margins = []
        for l in range(len(SPEC) + 1):
            C, Np, S = certificate(As[l], Bs[l])
            cap = min(r, dims[l])
            margins.append(int(torch.linalg.matrix_rank(C, rtol=1e-10)))
            if l < len(SPEC):
                Pt = kept[l]
                res = torch.linalg.norm(C @ Pt, dim=1) / (torch.linalg.norm(As[l] @ Pt, dim=1) + 1e-300)
                res_med = float(res.median()); P_l = int(Pt.shape[2])
            else:
                res_med = float("nan"); P_l = 1
            emit(dict(part="CONVLAYER", r=r, layer=l + 1, kind="conv" if l < len(SPEC) else "head",
                      d_l=dims[l], rank_cap=cap, n_prime=Np, certificate_margin=cap - Np,
                      rank_C=margins[-1], patches_per_image=P_l,
                      conditions_per_image=margins[-1] * P_l, cert_residual_median=res_med,
                      vacuous=bool(margins[-1] == 0), N=a.N, git=git_hash()))
            print(f"  r={r:4d} layer {l+1} ({'conv' if l < len(SPEC) else 'head'}): d={dims[l]} "
                  f"cap={cap} N'={Np} rank C={margins[-1]} x P={P_l} -> {margins[-1]*P_l} conditions/image"
                  f"{'  [VACUOUS]' if margins[-1] == 0 else ''}  cert residual {res_med:.2e}", flush=True)
        emit(dict(part="CONV_VERDICT", r=r, ranks_C=margins, any_margin=bool(any(x > 0 for x in margins)),
                  conv_margins=margins[:len(SPEC)], head_margin=margins[-1],
                  reading="CONV-CARRIES" if any(x > 0 for x in margins[:len(SPEC)]) else "CONV-VACUOUS",
                  seconds=time.time() - t0, backbone_acc=acc, N=a.N, T=a.T, lr=a.lr,
                  start_model="n/a (algebraic)", claim_class="algebraic check",
                  git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
