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
SPECS = {                                                           # (cin, cout, k, stride, pad) on 28x28
    # early-channel regime: few channels, many positions -> N*P swamps d and the patch span fills the layer
    "shallow": [(1, 32, 3, 2, 1), (32, 64, 3, 2, 1), (64, 64, 3, 2, 1)],
    # deep-channel regime (yoado-cd/81, audit 02b93e8): many channels, few positions -> N*P < d, so saturation
    # does NOT apply and the certificate should survive. d = 9, 576, 1152, 2304 against N*P = 1568, 392, 128, 32.
    "deep": [(1, 64, 3, 2, 1), (64, 128, 3, 2, 1), (128, 256, 3, 2, 1), (256, 256, 3, 2, 1)],
    # bottlenecked regime (plan 2026-09-18, Audit section): an 8-channel layer 3 (d_3 = 1152 -> d_4 = 8*9 = 72) so
    # the depth laws min_j(d_j + sum_{l<j} q_l) and min(k_1, sum q_l) separate; followed by a DENSE hidden layer
    # 1024 -> 1000 (GELU) before the 10-way head (DENSE_HIDDEN below).  Sides 28 -> 14 -> 7 -> 4 -> 2.
    "bottleneck": [(1, 64, 3, 2, 1), (64, 128, 3, 2, 1), (128, 8, 3, 2, 1), (8, 256, 3, 2, 1)],
}
# Optional dense GELU layer between the flattened conv stack and the head, by spec name (absent = none).  It is part
# of the FROZEN encoder: it has no LoRA slot, As[-1] still acts on the head's input, which is then its output.
DENSE_HIDDEN = {"bottleneck": 1000}
SPEC = SPECS["shallow"]


def patches_of(h, k, stride, pad):
    """(N, C, H, W) -> (N, C*k*k, P), the vectors a shared kernel actually multiplies."""
    return F.unfold(h, k, padding=pad, stride=stride)


def conv_forward(x, Wms, bs, Whead, bhead, As=None, Bs=None, want_patches=False, Wd=None, bd=None):
    """As/Bs may contain None entries: those layers are FROZEN, which is the point of the solo arm -- a frozen
       upstream means the adapted layer's input never moves, so its recorded count cannot be inflated by drift.
       Wd/bd: the optional frozen dense hidden layer (DENSE_HIDDEN) applied after the flatten, before the head."""
    """x: (N, 1, 28, 28).  Wms[l] is (cout, cin*k*k) -- the kernel as the matrix the certificate acts on."""
    h = x; kept = []
    for l, (cin, cout, k, s, p) in enumerate(SPEC):
        P = patches_of(h, k, s, p)
        if want_patches: kept.append(P)
        z = Wms[l] @ P + bs[l][:, None]
        if As is not None and As[l] is not None: z = z + Bs[l] @ (As[l] @ P)
        side = int(math.isqrt(z.shape[-1]))
        h = F.gelu(z.reshape(z.shape[0], cout, side, side))
    flat = h.reshape(h.shape[0], -1)
    if Wd is not None: flat = F.gelu(flat @ Wd.T + bd)
    z = flat @ Whead.T + bhead
    if As is not None and As[-1] is not None: z = z + (flat @ As[-1].T) @ Bs[-1].T
    return (z, kept) if want_patches else z


def head_input_dim(spec_name_or_list, dense_hidden=None):
    """Width of the head's input: the flattened conv stack, or the dense hidden layer when the spec has one."""
    spec = SPECS[spec_name_or_list] if isinstance(spec_name_or_list, str) else spec_name_or_list
    if dense_hidden is None and isinstance(spec_name_or_list, str): dense_hidden = DENSE_HIDDEN.get(spec_name_or_list)
    side = 28
    for _, _, k, st, pd in spec: side = (side + 2 * pd - k) // st + 1
    return dense_hidden if dense_hidden else spec[-1][1] * side * side


def run_training(x, Wms, bs, Whead, bhead, A0s, y, m, T, lr, Wd=None, bd=None):
    As = [None if a is None else a.detach().clone().requires_grad_(True) for a in A0s]
    Bs = [None if a is None else torch.zeros(o, a.shape[0], dtype=x.dtype, device=x.device, requires_grad=True)
          for o, a in zip([c for _, c, _, _, _ in SPEC] + [m], A0s)]
    live = [i for i, a in enumerate(As) if a is not None]
    Y = torch.eye(m, device=x.device)[y]
    for _ in range(T):
        z = conv_forward(x, Wms, bs, Whead, bhead, As, Bs, Wd=Wd, bd=bd)
        zs = z - z.max(dim=1, keepdim=True).values
        p = torch.exp(zs); p = p / p.sum(dim=1, keepdim=True)
        loss = -(Y * torch.log(p + 1e-300)).sum() / x.shape[0]
        params = [As[i] for i in live] + [Bs[i] for i in live]
        gs = torch.autograd.grad(loss, params)
        n = len(live)
        for j, i in enumerate(live):
            As[i] = (As[i] - lr * gs[j]).detach().requires_grad_(True)
            Bs[i] = (Bs[i] - lr * gs[n + j]).detach().requires_grad_(True)
    return ([None if a is None else a.detach() for a in As],
            [None if b is None else b.detach() for b in Bs])


def train_backbone(Xtr, ytr, Xte, yte, dev, epochs, bs, lr, seed, init=None, target_train_acc=None,
                   min_train_loss=None, plateau_patience=5, return_stats=False, dense_hidden=None):
    """Adam, no augmentation, no weight decay.  Positional use is byte-identical to the original.
       WP0 keywords (2026-09-18): `init` = a checkpoint dict of this format (Wms/bs/Whead/bhead) to warm-start
       from; `target_train_acc` / `min_train_loss` = stop at the first epoch whose FULL-train accuracy >= the
       first and mean cross-entropy <= the second (either alone if only one is given; lr halves after
       `plateau_patience` epochs without train-loss improvement, floor 1e-5); `return_stats` also returns a dict
       with train acc / loss / margin fraction, test acc / loss, epochs run and whether the rule was met;
       `dense_hidden` = width of a GELU dense layer between the flatten and the head (its Wd/bd are returned in
       the stats dict, so it requires return_stats=True)."""
    if dense_hidden and not return_stats: raise ValueError("dense_hidden needs return_stats=True (Wd/bd are returned there)")
    torch.manual_seed(seed)
    convs = nn.ModuleList([nn.Conv2d(ci, co, k, s, p) for ci, co, k, s, p in SPEC]).to(dev).float()
    side = 28
    for _, _, k, st, pd in SPEC: side = (side + 2 * pd - k) // st + 1
    flat_dim = SPEC[-1][1] * side * side
    dense = nn.Linear(flat_dim, dense_hidden).to(dev).float() if dense_hidden else None
    head = nn.Linear(dense_hidden or flat_dim, 10).to(dev).float()
    if init is not None:
        with torch.no_grad():
            for c, w, b in zip(convs, init["Wms"], init["bs"]):
                c.weight.copy_(w.reshape(c.weight.shape).float()); c.bias.copy_(b.float())
            head.weight.copy_(init["Whead"].float()); head.bias.copy_(init["bhead"].float())
            if dense is not None: dense.weight.copy_(init["Wd"].float()); dense.bias.copy_(init["bd"].float())
    params = list(convs.parameters()) + list(head.parameters()) + (list(dense.parameters()) if dense else [])
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.CrossEntropyLoss()
    def f(x):
        h = x
        for c in convs: h = F.gelu(c(h))
        h = h.reshape(h.shape[0], -1)
        if dense is not None: h = F.gelu(dense(h))
        return head(h)
    @torch.no_grad()
    def stats(X, y, chunk=2000):
        correct = 0; loss = 0.0; pos = 0
        for i in range(0, X.shape[0], chunk):
            z = f(X[i:i + chunk]); yb = y[i:i + chunk]
            loss += float(lossf(z, yb) * len(yb)); correct += int((z.argmax(1) == yb).sum())
            zy = z.gather(1, yb[:, None])[:, 0]; zo = z.clone(); zo.scatter_(1, yb[:, None], -float("inf"))
            pos += int((zy - zo.max(1).values > 0).sum())
        return dict(acc=correct / X.shape[0], loss=loss / X.shape[0], margin_pos_frac=pos / X.shape[0])
    rule = target_train_acc is not None or min_train_loss is not None
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=plateau_patience,
                                                        min_lr=1e-5) if rule else None)
    stopped_at = None; t0 = time.time()
    if rule:
        tr = stats(Xtr, ytr)
        print(f"  epoch 0  train acc {tr['acc']*100:.3f}%  train loss {tr['loss']:.3e}", flush=True)
    for ep in range(epochs):
        perm = torch.randperm(Xtr.shape[0], device=dev)
        for i in range(0, Xtr.shape[0], bs):
            j = perm[i:i + bs]
            opt.zero_grad(); lossf(f(Xtr[j]), ytr[j]).backward(); opt.step()
        with torch.no_grad():
            acc = float((f(Xte).argmax(1) == yte).double().mean())
        if rule:
            tr = stats(Xtr, ytr); sched.step(tr["loss"])
            print(f"  epoch {ep+1}  train acc {tr['acc']*100:.3f}%  train loss {tr['loss']:.3e}  test acc {acc*100:.2f}%  "
                  f"lr {opt.param_groups[0]['lr']:.1e}  {time.time()-t0:.0f}s", flush=True)
            if ((target_train_acc is None or tr["acc"] >= target_train_acc) and
                    (min_train_loss is None or tr["loss"] <= min_train_loss)):
                stopped_at = ep + 1; print(f"  stopping rule met at epoch {stopped_at}", flush=True); break
        else:
            print(f"  epoch {ep+1}  test acc {acc*100:.2f}%", flush=True)
    Wms = [c.weight.detach().double().reshape(c.weight.shape[0], -1) for c in convs]
    bs_ = [c.bias.detach().double() for c in convs]
    out = (Wms, bs_, head.weight.detach().double(), head.bias.detach().double(), acc)
    if not return_stats: return out
    tr = stats(Xtr, ytr); te = stats(Xte, yte)
    return out + (dict(train_acc=tr["acc"], train_loss=tr["loss"], train_margin_pos_frac=tr["margin_pos_frac"],
                       test_acc=te["acc"], test_loss=te["loss"], epochs_run=stopped_at or epochs,
                       rule_met=(stopped_at is not None) if rule else None,
                       lr_final=opt.param_groups[0]["lr"],
                       Wd=None if dense is None else dense.weight.detach().double(),
                       bd=None if dense is None else dense.bias.detach().double()),)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--ranks", nargs="*", type=int,
                    default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--T", type=int, default=200); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--backbone-lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--spec", default="shallow", choices=sorted(SPECS), help="shallow = early-channel regime "
                    "(N*P swamps d); deep = many channels, few positions (N*P < d), where saturation does NOT apply")
    ap.add_argument("--arms", nargs="*", default=["all"], help="'all' adapts every layer (inputs DRIFT); "
                    "'solo' adapts one layer at a time with everything else frozen, so the adapted layer's input "
                    "never moves -- this separates SATURATION (patch span fills d) from DRIFT (A_0 h leaves row B_T)")
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--ckpt", default="models/exact_inversion/mnist_conv.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    global SPEC
    SPEC = SPECS[a.spec]
    if a.ckpt == ap.get_default("ckpt") and a.spec != "shallow":
        a.ckpt = f"models/exact_inversion/mnist_conv_{a.spec}.pth"
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
        Wd = ck.get("Wd"); bd = ck.get("bd")
        print(f"# loaded conv backbone {a.ckpt} test acc {acc*100:.2f}%", flush=True)
    else:
        os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
        dh = DENSE_HIDDEN.get(a.spec)
        if dh:
            Wms, bs_, Whead, bhead, acc, st = train_backbone(Xtr_t, ytr_t, Xte_t, yte_t, dev, a.epochs, a.bs,
                                                             a.backbone_lr, a.seed, return_stats=True, dense_hidden=dh)
            Wd, bd = st["Wd"], st["bd"]
        else:
            Wms, bs_, Whead, bhead, acc = train_backbone(Xtr_t, ytr_t, Xte_t, yte_t, dev, a.epochs, a.bs,
                                                         a.backbone_lr, a.seed)
            Wd = bd = None
        torch.save(dict(Wms=[w.cpu() for w in Wms], bs=[b.cpu() for b in bs_], Whead=Whead.cpu(),
                        bhead=bhead.cpu(), test_acc=acc, git=git_hash(),
                        **({"Wd": Wd.cpu(), "bd": bd.cpu(), "spec": a.spec} if Wd is not None else {})), a.ckpt)
        print(f"# trained conv backbone -> {a.ckpt} test acc {acc*100:.2f}%", flush=True)
    Wms = [w.double() for w in Wms]; bs_ = [b.double() for b in bs_]
    Whead = Whead.double(); bhead = bhead.double()
    if Wd is not None: Wd = Wd.to(dev).double(); bd = bd.to(dev).double()

    g = torch.Generator().manual_seed(a.seed + 7)
    idx = torch.randperm(Xte_t.shape[0], generator=g)[:a.N].to(dev)
    X = Xte_t[idx].double(); y = yte_t[idx]
    m = 10
    side = 28
    for _, _, k, st, pd in SPEC: side = (side + 2 * pd - k) // st + 1
    dims = [ci * k * k for ci, co, k, s, p in SPEC] + [Whead.shape[1]]      # the head's input: flatten, or the dense layer

    # the count that decides everything, measured on the FROZEN backbone first: how many independent patch
    # directions do N recorded images actually supply at each layer?
    with torch.no_grad():
        _, kept = conv_forward(X, Wms, bs_, Whead, bhead, want_patches=True, Wd=Wd, bd=bd)
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

    arms = []
    for arm in a.arms:
        if arm == "all": arms.append(("all", list(range(len(SPEC) + 1))))
        elif arm == "solo": arms += [(f"solo{l+1}", [l]) for l in range(len(SPEC) + 1)]
        else: arms.append((arm, [int(x) - 1 for x in arm.split(",")]))
    for arm_name, adapted in arms:
     for r in a.ranks:
        sigma0 = a.sigma0 or 1.0 / math.sqrt(dims[0])
        gA = torch.Generator().manual_seed(a.seed + 11)
        A0s = [((sigma0 * torch.randn(r, d, generator=gA)).to(dev) if i in adapted else None)
               for i, d in enumerate(dims)]
        t0 = time.time()
        As, Bs = run_training(X, Wms, bs_, Whead, bhead, A0s, y, m, a.T, a.lr, Wd=Wd, bd=bd)
        with torch.no_grad():
            _, kept = conv_forward(X, Wms, bs_, Whead, bhead, As, Bs, want_patches=True, Wd=Wd, bd=bd)
            h = X
            for l, (cin, cout, k, s, p) in enumerate(SPEC):
                Pt = patches_of(h, k, s, p)
                z = Wms[l] @ Pt + bs_[l][:, None]
                if As[l] is not None: z = z + Bs[l] @ (As[l] @ Pt)
                side = int(math.isqrt(z.shape[-1])); h = F.gelu(z.reshape(z.shape[0], cout, side, side))
            head_in = h.reshape(h.shape[0], -1)
            if Wd is not None: head_in = F.gelu(head_in @ Wd.T + bd)   # through the frozen dense layer
            head_in = head_in.T                                        # (d_head, N) -- the head's own inputs
        margins = []; conv_holds = []
        for l in range(len(SPEC) + 1):
            if As[l] is None:
                margins.append(0)
                if l < len(SPEC): conv_holds.append(False)
                continue
            C, Np, S = certificate(As[l], Bs[l])
            cap = min(r, dims[l])
            # rank(C) needs an ABSOLUTE floor set by A_T, not a floor relative to C's own largest singular value:
            # when the projector annihilates A_T the surviving matrix is ~1e-16 x A_T, its singular values are all
            # tiny but comparable, and a relative rank call returns FULL rank for a matrix that is zero.
            svC = torch.linalg.svdvals(C); sA = float(torch.linalg.svdvals(As[l])[0])
            margins.append(int((svC > 1e-10 * sA).sum()))
            if l < len(SPEC):
                Pt = kept[l]
                res = torch.linalg.norm(C @ Pt, dim=1) / (torch.linalg.norm(As[l] @ Pt, dim=1) + 1e-300)
                res_med = float(res.median()); P_l = int(Pt.shape[2])
            else:                                                  # the head: its input is the flattened stack
                res = torch.linalg.norm(C @ head_in, dim=0) / (torch.linalg.norm(As[l] @ head_in, dim=0) + 1e-300)
                res_med = float(res.median()); P_l = 1
            # a margin is worth nothing unless the condition actually HOLDS at the truth: with every layer
            # adapted the features drift, A_0 h_i need not lie in row(B_T), and C h is then not zero at all.
            holds = bool(res_med == res_med and res_med < 1e-8)
            if l < len(SPEC): conv_holds.append(holds)
            emit(dict(part="CONVLAYER", arm=arm_name, spec=a.spec, adapted=[i + 1 for i in adapted], r=r, layer=l + 1, kind="conv" if l < len(SPEC) else "head",
                      d_l=dims[l], rank_cap=cap, n_prime=Np, certificate_margin=cap - Np,
                      rank_C=margins[-1], patches_per_image=P_l, rank_B_capped_by_width=bool(Np >= min(r, Bs[l].shape[0])),
                      conditions_per_image=margins[-1] * P_l, cert_residual_median=res_med,
                      certificate_holds_at_truth=holds, usable=bool(margins[-1] > 0 and holds),
                      vacuous=bool(margins[-1] == 0), N=a.N, git=git_hash()))
            print(f"  [{arm_name}] r={r:4d} layer {l+1} ({'conv' if l < len(SPEC) else 'head'}): d={dims[l]} "
                  f"cap={cap} N'={Np} rank C={margins[-1]} x P={P_l} -> {margins[-1]*P_l} conditions/image"
                  f"{'  [VACUOUS]' if margins[-1] == 0 else ''}  cert residual {res_med:.2e}", flush=True)
        usable_conv = [l for l in range(len(SPEC)) if margins[l] > 0 and conv_holds[l]]
        emit(dict(part="CONV_VERDICT", arm=arm_name, spec=a.spec, adapted=[i + 1 for i in adapted], r=r, ranks_C=margins, any_margin=bool(any(x > 0 for x in margins)),
                  conv_margins=margins[:len(SPEC)], head_margin=margins[-1],
                  conv_layers_with_margin=[l + 1 for l in range(len(SPEC)) if margins[l] > 0],
                  conv_layers_usable=[l + 1 for l in usable_conv],
                  reading="CONV-CARRIES" if usable_conv else "CONV-VACUOUS",
                  reading_note="CARRIES requires a POSITIVE margin AND the condition holding at the truth; a "
                               "margin that exists only because rank B_T is capped by the output width, with "
                               "C h nowhere near zero, is not a certificate",
                  seconds=time.time() - t0, backbone_acc=acc, N=a.N, T=a.T, lr=a.lr,
                  start_model="n/a (algebraic)", claim_class="algebraic check",
                  git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
