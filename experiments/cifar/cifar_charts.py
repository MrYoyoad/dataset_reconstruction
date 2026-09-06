"""
CIFAR certificate replica: chart / layer / solver variants (cifar_certificate.py at the repo root is left untouched).

The raw replica (jobs 252897/252898) and its on-chart control (257893/257895) fail to land: LoRA on the FIRST layer
makes C act LINEARLY on pixels, the search used a conv-AE decoder chart with a batched Adam descent, and the residual's
minima on the chart are spurious (one collapsed mean apple at k=32; noise at k=48). This script varies the three things
that differ from the MNIST cell that lands (RESULTS Steps 22-25: head LoRA on FEATURES, PCA chart, Levenberg-Marquardt):

  --layer 1|2|3    1: adapt W1 (input = pixels, C LINEAR on pixels: every blend of the privates is an exact zero -- audit
                   2026-09-06, so no chart can separate them).  2: adapt W2 (input = h1, 512-dim, one GELU).  3: adapt the
                   HEAD W3 (input = penultimate features, 256-dim, two GELUs) -- the literal Figure-2 setting (head LoRA on
                   features; --zero_row zeroes the 11th head row as in the letters cell).
  --wrong_release  CONTROL: train the release on 8 OTHER public apples (same A0, same chart), score landings against the
                   original privates -- must be 0.
  --chart pca|ae   PCA on public images (linear chart, mean + V z) or the conv-AE decoder of cifar_certificate.py.
  --solver adam|lm Adam 3000 iterations batched (as the replica) or Levenberg-Marquardt 300 iterations per start
                   (certificate.py's lm_cert, autograd Jacobian) as on MNIST.
  --private raw|onchart   private training inputs = the raw images, or their chart projections (exact zeros on the chart).
  --chart oracle --eps E   NOT attacker-available: span of the privates + eps-noise (relative), PCA filler -- the ladder that measures
                   the representation error below which raw privates land (RESULTS Step 11 'exact' chart analogue).
  --obj repo|cn    ||C phi|| / ||A_T phi|| (the repo's audited scale-invariant objective) or ||C phi|| / (||C|| ||phi||).

Same seed, data split, base model and AE construction order as cifar_certificate.py (the layer-2/3 A0 comes from a separate
generator), so the base model and the AE are the ones of jobs 252897/252898. Everything in float64 except the AE (fp32-trained, evaluated in float64).

  python experiments/cifar/cifar_charts.py --layer 2 --chart pca --solver lm --private onchart --k 32 --out experiments/cifar/charts/L2_pca_lm_on_k32
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as tf

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=8); p.add_argument("--r", type=int, default=64); p.add_argument("--k", type=int, default=32)
p.add_argument("--T", type=int, default=200); p.add_argument("--lr", type=float, default=0.05); p.add_argument("--cls100", type=int, default=0)
p.add_argument("--starts", type=int, default=400); p.add_argument("--pre_epochs", type=int, default=15); p.add_argument("--ae_epochs", type=int, default=40)
p.add_argument("--seed", type=int, default=0); p.add_argument("--out", type=str, required=True)
p.add_argument("--layer", type=int, choices=[1, 2, 3], default=3); p.add_argument("--zero_row", action="store_true"); p.add_argument("--wrong_release", action="store_true"); p.add_argument("--chart", choices=["pca", "ae", "oracle"], default="pca"); p.add_argument("--eps", type=float, default=0.0, help="oracle chart: noise level added to the privates before spanning (relative to ||x_i - mean||)")
p.add_argument("--solver", choices=["adam", "lm"], default="lm"); p.add_argument("--private", choices=["raw", "onchart"], default="onchart")
p.add_argument("--obj", choices=["repo", "cn"], default="repo"); p.add_argument("--lm_iters", type=int, default=300); p.add_argument("--adam_iters", type=int, default=3000)
p.add_argument("--start_scale", type=float, default=1.0, help="multiplier on the public coordinate std of the random starts")
args = p.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out, exist_ok=True)
D = 3 * 32 * 32
def log(s): print(s, flush=True)

# ---------------------------------------------------------------- data (identical to cifar_certificate.py)
T10 = tf.ToTensor()
c10_tr = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=T10)
c10_te = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=T10)
c100_tr = torchvision.datasets.CIFAR100("data", train=True, download=True, transform=T10)
def stack(ds, idx): return torch.stack([ds[i][0] for i in idx]).reshape(len(idx), -1)
idx100 = [i for i, y in enumerate(c100_tr.targets) if y == args.cls100]
perm = np.random.permutation(len(idx100))
priv_idx = [idx100[i] for i in perm[:args.N]]; pub_idx = [idx100[i] for i in perm[args.N:]]
H_raw = stack(c100_tr, priv_idx); X_pub = stack(c100_tr, pub_idx)
log(f"private {H_raw.shape}, public same-class {X_pub.shape}")

# ---------------------------------------------------------------- base model (identical)
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.l1 = nn.Linear(D, 512); s.l2 = nn.Linear(512, 256); s.l3 = nn.Linear(256, 11)
    def forward(s, x, W1=None, W2=None, W3=None):
        h = F.gelu(F.linear(x, s.l1.weight if W1 is None else W1, s.l1.bias))
        h = F.gelu(F.linear(h, s.l2.weight if W2 is None else W2, s.l2.bias))
        return F.linear(h, s.l3.weight if W3 is None else W3, s.l3.bias)
    def h1(s, x): return F.gelu(F.linear(x, s.l1.weight, s.l1.bias))
    def h2(s, x): return F.gelu(F.linear(s.h1(x), s.l2.weight, s.l2.bias))
base = MLP().to(dev); opt = torch.optim.Adam(base.parameters(), 1e-3)
ld = torch.utils.data.DataLoader(c10_tr, batch_size=256, shuffle=True, num_workers=2)
for ep in range(args.pre_epochs):
    for x, y in ld:
        x = x.reshape(len(x), -1).to(dev); y = y.to(dev); opt.zero_grad(); F.cross_entropy(base(x), y).backward(); opt.step()
with torch.no_grad():
    xt = torch.stack([c10_te[i][0] for i in range(2000)]).reshape(2000, -1).to(dev); yt = torch.tensor(c10_te.targets[:2000]).to(dev)
    acc = (base(xt).argmax(1) == yt).float().mean().item(); log(f"base CIFAR-10 acc (2k test) = {acc:.3f}")
base64 = MLP().to(dev).double(); base64.load_state_dict(base.state_dict())
for q_ in base64.parameters(): q_.requires_grad_(False)
if args.zero_row:
    with torch.no_grad(): base64.l3.weight[10].zero_(); base64.l3.bias[10].zero_()
W0 = {1: base64.l1.weight, 2: base64.l2.weight, 3: base64.l3.weight}[args.layer].detach()
n_in = W0.shape[1]
phi = {1: (lambda x: x), 2: (lambda x: base64.h1(x)), 3: (lambda x: base64.h2(x))}[args.layer]      # the adapted layer's INPUT as a function of pixels
def fwd(x, W): return base64(x, **{f"W{args.layer}": W})

# ---------------------------------------------------------------- LoRA + certificate as a function of the private pixels
A0 = torch.randn(args.r, D, dtype=torch.float64, device=dev) / math.sqrt(D)       # the replica's draw, consumed so the AE stream is identical
if args.layer != 1:                                                                # separate generator: the global stream (and the AE) stay the replica's
    A0 = torch.randn(args.r, n_in, generator=torch.Generator().manual_seed(args.seed + 7), dtype=torch.float64).to(dev) / math.sqrt(n_in)

def lora_and_certificate(Xd):
    A = A0.clone().requires_grad_(True); B = torch.zeros(W0.shape[0], args.r, dtype=torch.float64, device=dev, requires_grad=True)
    y_priv = torch.full((args.N,), 10, device=dev)
    for t in range(args.T):
        loss = F.cross_entropy(fwd(Xd, W0 + B @ A), y_priv); gA, gB = torch.autograd.grad(loss, [A, B])
        with torch.no_grad(): A -= args.lr * gA; B -= args.lr * gB
    A_T, B_T = A.detach(), B.detach(); Hd = phi(Xd).detach()
    _, sB, VhB = torch.linalg.svd(B_T, full_matrices=False)
    gap = (sB[args.N - 1] / sB[args.N]).item()
    Vq = VhB[: args.N].T; P_perp = torch.eye(args.r, dtype=torch.float64, device=dev) - Vq @ Vq.T; C = P_perp @ A_T
    s1 = (C @ Hd.T).norm().item() / (C.norm().item() * Hd.norm().item()); s2 = torch.linalg.matrix_rank(C, rtol=1e-10).item()
    s3 = (C - P_perp @ A0).norm().item() / C.norm().item()
    log(f"LoRA loss after {args.T} steps: {loss.item():.3e};  sigma(B_T)[:N+2] = {[f'{v:.2e}' for v in sB[: args.N + 2].tolist()]}")
    log(f"gap sigma_N/sigma_N+1 = {gap:.2e}   ||CH||/(||C|| ||H||) = {s1:.2e}   rank C = {s2} (expect {args.r - args.N})   ||C - P_perp A0||/||C|| = {s3:.2e}")
    return A_T, B_T, C, dict(gap=gap, CH_rel=s1, rank_C=s2, quotient=s3, loss=loss.item())

# ---------------------------------------------------------------- charts
X_chart = torch.cat([X_pub, stack(c10_tr, list(range(0, 50000, 5)))]).to(dev)
class AE(nn.Module):
    def __init__(s, k):
        super().__init__()
        s.enc = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
                              nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Flatten(), nn.Linear(128 * 16, k))
        s.dec_fc = nn.Linear(k, 128 * 16)
        s.dec = nn.Sequential(nn.GELU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
                              nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())
    def G(s, z): return s.dec(s.dec_fc(z).view(-1, 128, 4, 4)).reshape(len(z), -1)
    def forward(s, x): return s.G(s.enc(x.view(-1, 3, 32, 32)))
if args.chart == "ae":
    ae = AE(args.k).to(dev); opt = torch.optim.Adam(ae.parameters(), 1e-3)
    for ep in range(args.ae_epochs):
        for i in torch.randperm(len(X_chart)).split(256):
            xb = X_chart[i]; opt.zero_grad(); F.mse_loss(ae(xb), xb).backward(); opt.step()
    ae.eval(); ae.double()
    for q_ in ae.parameters(): q_.requires_grad_(False)
    G = lambda z: ae.G(z.reshape(-1, args.k))                     # [B, k] -> [B, 3072]
    enc = lambda x: ae.enc(x.reshape(-1, 3, 32, 32).double())
    with torch.no_grad(): coord_std = enc(X_chart[:5000]).std(0)
else:
    Xc = X_chart.double(); mean = Xc.mean(0); U_, S_, Vh_ = torch.linalg.svd(Xc - mean, full_matrices=False); V = Vh_[: args.k].T.contiguous()
    explained = float((S_[: args.k] ** 2).sum() / (S_ ** 2).sum()); log(f"PCA chart k={args.k}: explained variance {explained:.3f}")
    if args.chart == "oracle":                                                    # NOT ATTACKER-AVAILABLE: the epsilon-oracle ladder (RESULTS Step 11 'exact' chart analogue)
        go = torch.Generator().manual_seed(args.seed + 99); Xd = H_raw.double().to(dev) - mean
        noise = torch.randn(Xd.shape, generator=go, dtype=torch.float64).to(dev); noise = noise / noise.norm(dim=1, keepdim=True) * Xd.norm(dim=1, keepdim=True) * args.eps
        Q_, _ = torch.linalg.qr(torch.cat([(Xd + noise).T, V[:, : args.k - args.N]], 1)); V = Q_[:, : args.k].contiguous()
        log(f"ORACLE chart k={args.k}, eps={args.eps}: span of the noised privates + {args.k - args.N} PCA components (not attacker-available)")
    G = lambda z: mean + z.reshape(-1, args.k) @ V.T
    enc = lambda x: (x.double() - mean) @ V
    with torch.no_grad(): coord_std = enc(X_chart[:5000]).std(0)
proj = lambda x: G(enc(x))

# ---------------------------------------------------------------- private set for this run, release, certificate
Hf_raw = H_raw.double().to(dev)
with torch.no_grad(): H_proj = proj(Hf_raw)                    # NOT clamped: the on-chart training input must be exactly G(z)
Xtrain = H_proj if args.private == "onchart" else Hf_raw                      # the ORIGINAL privates: landings are always scored against these
X_release = Xtrain
if args.wrong_release:                                                        # control: the release comes from 8 OTHER apples
    others = X_pub[args.N: 2 * args.N].double().to(dev)
    with torch.no_grad(): X_release = proj(others) if args.private == "onchart" else others
log(f"=== layer {args.layer}, chart {args.chart}, solver {args.solver}, private {args.private}, obj {args.obj}, k {args.k}, zero_row {args.zero_row}, wrong_release {args.wrong_release} ===")
A_T, B_T, C, san = lora_and_certificate(X_release)
with torch.no_grad(): Hpriv = phi(Xtrain)                                     # features of the original privates (blend diagnostic)
def blend_fraction(x):
    """fraction of ||phi(x)||^2 explained by the least-squares fit in span{phi(x_i)}: ~1 = a blend of the privates (audit)."""
    f = phi(x); Q, _ = torch.linalg.qr(Hpriv.T); proj_f = (f @ Q) @ Q.T
    return (proj_f.norm(dim=1) ** 2 / f.norm(dim=1) ** 2)

import kornia
def ssim(a, b):
    a = a.reshape(-1, 3, 32, 32).float().clamp(0, 1); b = b.reshape(-1, 3, 32, 32).float().clamp(0, 1)
    return kornia.metrics.ssim(a, b, window_size=3).mean(dim=(1, 2, 3))

def objective_vec(x):                                             # x: [B, 3072] -> per-image residual VECTORS [B, r] (scaled)
    f = phi(x); Cf = f @ C.T
    if args.obj == "repo": return Cf / torch.linalg.norm(f @ A_T.T, dim=1, keepdim=True)
    return Cf / (C.norm() * torch.linalg.norm(f, dim=1, keepdim=True))
def residual(x): return torch.linalg.norm(objective_vec(x), dim=1)

with torch.no_grad():
    res_truth = residual(Xtrain); res_proj = residual(H_proj); floor_ssim = ssim(H_proj, Xtrain); floor_ssim_raw = ssim(H_proj, Hf_raw)
    Xp = X_pub.double().to(dev); Xp_proj = proj(Xp[:200])
    ctrl = Xp_proj[torch.cdist(H_proj, Xp_proj).argmin(1)]                              # control = the PROJECTION of the public apple nearest each private's projection (audit: window-3 SSIM rewards blur)
    feat_ref = float(torch.linalg.norm(phi(Xp[:200]) @ A_T.T, dim=1).median())
    blend_truth = blend_fraction(Xtrain); blend_public = blend_fraction(Xp[:200])
    # spurious-zero reference: the residual on public images (same class, never trained on) and on random chart points
    res_public = residual(X_pub[:200].double().to(dev)); res_chart_rand = residual(G(torch.randn(200, args.k, device=dev, dtype=torch.float64) * coord_std))
log(f"residual at the private training inputs: {[f'{v:.1e}' for v in res_truth.tolist()]}")
log(f"residual at the chart projections       : {[f'{v:.1e}' for v in res_proj.tolist()]}")
log(f"blend fraction: privates {[f'{v:.3f}' for v in blend_truth.tolist()]}, public median {blend_public.median():.3f}")
log(f"residual on 200 public same-class images: median {res_public.median():.2e}, min {res_public.min():.2e};  on 200 random chart points: median {res_chart_rand.median():.2e}, min {res_chart_rand.min():.2e}")

# ---------------------------------------------------------------- search
def lm_search(z0, iters, lam=1e-2):
    """certificate.lm_cert on the map z -> objective_vec(G(z)), one start; Jacobian by forward-mode autograd (reverse-mode fallback)."""
    import torch.func as tfn
    fun = lambda z: objective_vec(G(z)).reshape(-1)
    try: jac = lambda z: tfn.jacfwd(fun)(z)
    except Exception: jac = lambda z: torch.autograd.functional.jacobian(fun, z)
    z = z0.clone(); f = fun(z); obj = float(f @ f)
    for it in range(iters):
        try: J = jac(z)
        except Exception: jac = lambda z: torch.autograd.functional.jacobian(fun, z); J = jac(z)
        for _ in range(12):
            step = torch.linalg.solve(J.T @ J + lam * torch.eye(J.shape[1], device=z.device, dtype=z.dtype), -(J.T @ f))
            z_new = z + step; f_new = fun(z_new); obj_new = float(f_new @ f_new)
            if obj_new < obj: z, f, obj = z_new, f_new, obj_new; lam = max(lam / 3, 1e-15); break
            lam *= 4
        else: return z, obj, it + 1
        if obj < 1e-30: return z, obj, it + 1
    return z, obj, iters

t0 = time.time(); gz = torch.Generator(device="cpu").manual_seed(args.seed + 31)
Z0 = torch.randn(args.starts, args.k, generator=gz).to(dev).double() * coord_std * args.start_scale
if args.solver == "adam":
    z = Z0.clone().requires_grad_(True); opt = torch.optim.Adam([z], 5e-2)
    for it in range(args.adam_iters):
        opt.zero_grad(); R = residual(G(z)); (R ** 2).sum().backward(); opt.step()
        if it % 500 == 0: log(f"  it {it:4d}  median residual {R.median().item():.2e}  min {R.min().item():.2e}")
    with torch.no_grad(): Z = z.detach(); R = residual(G(Z)); iters = [args.adam_iters] * args.starts
else:
    Z = torch.zeros_like(Z0); R = torch.zeros(args.starts, device=dev, dtype=torch.float64); iters = []
    for s in range(args.starts):
        zs, obj, it = lm_search(Z0[s], args.lm_iters); Z[s] = zs.detach(); R[s] = math.sqrt(max(obj, 0.0)); iters.append(it)
        if (s + 1) % 50 == 0: log(f"  {s + 1}/{args.starts} starts, {time.time() - t0:.0f}s, median residual so far {R[: s + 1].median().item():.2e}, min {R[: s + 1].min().item():.2e}")
with torch.no_grad():
    X_found = G(Z)                                                # unclamped (errors are measured against the unclamped chart point)
    feat_ratio = torch.linalg.norm(phi(X_found) @ A_T.T, dim=1) / feat_ref; degenerate = feat_ratio < 0.05   # certificate.py's 0/0 guard
    R = torch.where(degenerate, torch.full_like(R, float("inf")), R)                                          # degenerate starts never rank
log(f"search done in {time.time() - t0:.0f}s")

with torch.no_grad():
    err = torch.stack([(X_found - Xtrain[i:i + 1]).norm(dim=1) / Xtrain[i:i + 1].norm() for i in range(args.N)], 1)  # [starts, N]
    nearest = err.argmin(1); err_near = err.min(1).values; landed = err_near < 1e-2
    S = torch.stack([ssim(X_found, Xtrain[i:i + 1].expand_as(X_found)) for i in range(args.N)], 1)
    best_idx = err.argmin(0)                                                   # closest approach per private image
    best_ssim = torch.stack([S[best_idx[i], i] for i in range(args.N)])
    raw_ssim = torch.stack([ssim(X_found[best_idx[i]:best_idx[i] + 1], Hf_raw[i:i + 1]) for i in range(args.N)]).squeeze()
    ctrl_ssim = torch.stack([ssim(X_found, ctrl[i:i + 1].expand_as(X_found)).max() for i in range(args.N)])   # same max-over-starts statistic
    blend_found = blend_fraction(X_found); blend_best = blend_found[best_idx]
    pd = torch.cdist(X_found, X_found); collapse = (pd < 1e-2 * X_found.norm(dim=1).mean()).double().mean().item()
    per_img = [int((landed & (nearest == i)).sum()) for i in range(args.N)]
    # attacker's view: rank starts by residual; how many of the lowest-residual starts are landings?
    order = torch.argsort(R); top = [int(landed[j]) for j in order[:20].tolist()]; n_degenerate = int(degenerate.sum())
# ---- ISOLATION TEST (attacker-side; release and chart only, no ground truth). At a returned point the rank of
# C times the chart Jacobian decides local isolation: FULL column rank (k) certifies the zero is isolated. Rank
# deficiency does NOT certify non-isolation -- a degenerate isolated zero is also rank-deficient -- so this is
# read ONE-SIDED, in the positive direction only (yoado-c6's correction to the test as first offered).
import torch.func as tfn
def isolation_rank(z1):
    f = lambda w: (phi(G(w.reshape(1, -1))) @ C.T).reshape(-1)
    sv = torch.linalg.svdvals(tfn.jacrev(f)(z1).detach())
    return int((sv > 1e-10 * sv[0]).sum()), float(sv[-1] / sv[0])
# CHART-JACOBIAN CONDITIONING, the quantity that separates a chart's CAPACITY from its usability by a search.
# For PCA the decoder is affine with orthonormal columns, so its Jacobian is an isometry and cond = 1 at every
# latent. For a learned decoder the conditioning varies with the latent, and a search stepping in latent space is
# then badly scaled wherever it is large -- which more training need not fix.
def chart_cond(z1):
    J = tfn.jacrev(lambda w: G(w.reshape(1, -1)).reshape(-1))(z1).detach()
    sv = torch.linalg.svdvals(J)
    return float(sv[0] / sv[-1].clamp(min=1e-300)), float(sv[-1])
z_truth = enc(H_proj) if args.chart == "ae" else enc(H_proj)
cond_truth = [chart_cond(z_truth[i]) for i in range(args.N)]
cond_found = [chart_cond(Z[j]) for j in order[:20].tolist()]
print(f"chart Jacobian conditioning at the private latents: cond {[f'{c:.1f}' for c, _ in cond_truth]}")
print(f"chart Jacobian conditioning at the top-20 starts   : median cond {float(np.median([c for c, _ in cond_found])):.1f}")
iso_best = [isolation_rank(Z[best_idx[i]]) for i in range(args.N)]
iso_top = [isolation_rank(Z[j]) for j in order[:20].tolist()]
print(f"isolation test at the closest start per image: ranks {[r for r, _ in iso_best]} of k={args.k} "
      f"(full rank certifies an isolated zero; deficient rank certifies nothing)")
print(f"isolation test on the top-20 by residual: {sum(1 for r, _ in iso_top if r == args.k)}/20 at full rank")

res = dict(layer=args.layer, chart=args.chart, solver=args.solver, private=args.private, obj=args.obj, k=args.k, r=args.r, N=args.N, T=args.T, lr=args.lr,
           seed=args.seed, starts=args.starts, base_acc=acc, zero_row=args.zero_row, wrong_release=args.wrong_release, eps=args.eps, n_degenerate=n_degenerate,
           blend_fraction_privates=blend_truth.tolist(), blend_fraction_public_median=float(blend_public.median()),
           blend_fraction_found_median=float(blend_found.median()), blend_fraction_best_per_image=blend_best.tolist(), sanity=san, residual_at_truth=res_truth.tolist(), residual_at_projection=res_proj.tolist(),
           residual_public_median=float(res_public.median()), residual_public_min=float(res_public.min()),
           residual_chart_random_median=float(res_chart_rand.median()), residual_chart_random_min=float(res_chart_rand.min()),
           landed=int(landed.sum()), landings_per_image=per_img, images_found=int(sum(1 for c in per_img if c > 0)),
           min_err_per_image=err.min(0).values.tolist(), best_ssim_vs_train=best_ssim.tolist(), best_ssim_vs_raw=raw_ssim.tolist(),
           chart_floor_ssim=floor_ssim.tolist(), chart_floor_ssim_vs_raw=floor_ssim_raw.tolist(), control_ssim=ctrl_ssim.tolist(), residual_best_per_image=[float(R[j]) for j in best_idx.tolist()],
           residual_min=float(R.min()), residual_median=float(R.median()), collapse=collapse, top20_by_residual_landed=top,
           chart_cond_at_truths=[c for c, _ in cond_truth], chart_cond_at_top20_median=float(np.median([c for c, _ in cond_found])),
           chart_sigma_min_at_truths=[v for _, v in cond_truth], ae_epochs=args.ae_epochs,
           isolation_rank_per_image=[r for r, _ in iso_best], isolation_smin_ratio_per_image=[v for _, v in iso_best],
           isolation_full_rank_top20=sum(1 for r, _ in iso_top if r == args.k),
           isolation_note="one-sided: full column rank (k) certifies the zero is isolated; deficient rank certifies nothing",
           lm_iters_median=float(np.median(iters)), seconds=time.time() - t0, host=socket.gethostname(), cmd=" ".join(sys.argv))
log("\n=== RESULT ===")
log(f"landed {res['landed']}/{args.starts}; per image {per_img}; images found {res['images_found']}/{args.N}; collapse {collapse:.2f}")
log(f"min rel. error per image : {[f'{v:.1e}' for v in res['min_err_per_image']]}")
log(f"best SSIM vs train input : {[f'{v:.2f}' for v in res['best_ssim_vs_train']]}")
log(f"best SSIM vs raw image   : {[f'{v:.2f}' for v in res['best_ssim_vs_raw']]}")
log(f"chart floor SSIM (vs train / vs raw): {[f'{v:.2f}' for v in res['chart_floor_ssim']]} / {[f'{v:.2f}' for v in res['chart_floor_ssim_vs_raw']]}")
log(f"control SSIM (max over starts vs nearest public apple): {[f'{v:.2f}' for v in res['control_ssim']]}")
log(f"blend fraction of found images: median {res['blend_fraction_found_median']:.3f}; closest starts {[f'{v:.2f}' for v in res['blend_fraction_best_per_image']]}; public median {res['blend_fraction_public_median']:.3f}")
log(f"residual of closest start: {[f'{v:.1e}' for v in res['residual_best_per_image']]}   (truth {res_truth.max():.1e}, min over starts {R.min():.1e}, median {R.median():.1e})")
log(f"top-20 starts by residual, landed? {top}   (degenerate starts excluded: {n_degenerate})")
print(json.dumps(res), flush=True)
with open(os.path.join(args.out, "result.json"), "w") as f: json.dump(res, f)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [("private (raw)", Hf_raw), ("training input (chart projection)" if args.private == "onchart" else "chart projection of the truth", H_proj),
        ("closest start (certificate only)", X_found[best_idx])]
fig, ax = plt.subplots(len(rows), args.N, figsize=(1.3 * args.N, 1.5 * len(rows)))
for i, (name, imgs) in enumerate(rows):
    for j in range(args.N):
        ax[i, j].imshow(imgs[j].reshape(3, 32, 32).permute(1, 2, 0).cpu().float().clamp(0, 1)); ax[i, j].axis("off")
        if i == 2: ax[i, j].set_title(f"{'landed' if per_img[j] else 'err %.2f' % res['min_err_per_image'][j]}, ssim {res['best_ssim_vs_train'][j]:.2f}", fontsize=6)
    ax[i, 0].text(-0.1, 1.15, name, fontsize=7, transform=ax[i, 0].transAxes)
fig.suptitle(f"CIFAR apples as 11th class: layer {args.layer}, chart {args.chart} k={args.k}, {args.solver}, private {args.private} -- {res['landed']}/{args.starts} landed, {res['images_found']}/{args.N} found", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(args.out, "grid.png"), dpi=150)
torch.save({"A_T": A_T.cpu(), "B_T": B_T.cpu(), "A0": A0.cpu(), "C": C.cpu(), "X_train": Xtrain.cpu(), "H_raw": H_raw, "H_proj": H_proj.cpu(), "priv_idx": priv_idx,
            "X_found": X_found.cpu(), "R": R.cpu(), "Z": Z.cpu(), "Z0": Z0.cpu(), "result": res}, os.path.join(args.out, "release_and_search.pt"))
log(f"saved {args.out}/grid.png, result.json, release_and_search.pt")
