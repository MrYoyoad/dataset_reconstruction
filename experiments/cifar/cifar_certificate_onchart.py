"""
On-chart CONTROL for cifar_certificate.py (repo root; that file is left untouched).

The raw run (jobs 252897 / 252898) passed every certificate sanity check but the search reached residuals BELOW the
chart floor on images that are not the private ones (k=32: every start collapsed onto one mean apple; k=48: off-manifold
noise). That is the raw-truth situation: the private images are not on the decoder chart, so the exact zero set of C
does not intersect the chart and the residual's minimiser on the chart is a spurious point. This control repeats the
experiment with the private training inputs REPLACED by their chart projections G(enc(x_i)) -- the exact zero set then
lies on the chart, as in the MNIST "on-chart" cells -- with the SAME seed, base model and autoencoder as the raw run
(same construction order, so every random stream is identical up to the LoRA step). It also saves every found image and
its residual, and a collapse diagnostic (pairwise distances among the found images).

  python cifar_certificate_onchart.py --N 8 --r 64 --k 32 --T 200 --starts 400 --out experiments/cifar/k32_onchart
"""
import argparse, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as tf

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=8); p.add_argument("--r", type=int, default=64); p.add_argument("--k", type=int, default=32)
p.add_argument("--T", type=int, default=200); p.add_argument("--lr", type=float, default=0.05); p.add_argument("--cls100", type=int, default=0)
p.add_argument("--starts", type=int, default=400); p.add_argument("--pre_epochs", type=int, default=15); p.add_argument("--ae_epochs", type=int, default=40)
p.add_argument("--seed", type=int, default=0); p.add_argument("--out", type=str, default="cifar_cert_onchart_out")
p.add_argument("--private", choices=["raw", "onchart"], default="onchart")
args = p.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out, exist_ok=True)
D = 3 * 32 * 32

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
print(f"private {H_raw.shape}, public same-class {X_pub.shape}")

# ---------------------------------------------------------------- base model (identical)
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.l1 = nn.Linear(D, 512); s.l2 = nn.Linear(512, 256); s.l3 = nn.Linear(256, 11)
    def forward(s, x, W1=None):
        h = F.gelu(F.linear(x, s.l1.weight if W1 is None else W1, s.l1.bias)); h = F.gelu(s.l2(h)); return s.l3(h)
base = MLP().to(dev); opt = torch.optim.Adam(base.parameters(), 1e-3)
ld = torch.utils.data.DataLoader(c10_tr, batch_size=256, shuffle=True, num_workers=2)
for ep in range(args.pre_epochs):
    for x, y in ld:
        x = x.reshape(len(x), -1).to(dev); y = y.to(dev); opt.zero_grad(); F.cross_entropy(base(x), y).backward(); opt.step()
with torch.no_grad():
    xt = torch.stack([c10_te[i][0] for i in range(2000)]).reshape(2000, -1).to(dev); yt = torch.tensor(c10_te.targets[:2000]).to(dev)
    print(f"base CIFAR-10 acc (2k test) = {(base(xt).argmax(1) == yt).float().mean().item():.3f}")
base64 = MLP().to(dev).double(); base64.load_state_dict(base.state_dict())
for q_ in base64.parameters(): q_.requires_grad_(False)
W0 = base64.l1.weight.detach()

# ---------------------------------------------------------------- LoRA + certificate, as a function of the private inputs
def lora_and_certificate(Hd, A0):
    A = A0.clone().requires_grad_(True); B = torch.zeros(512, args.r, dtype=torch.float64, device=dev, requires_grad=True)
    y_priv = torch.full((args.N,), 10, device=dev)
    with torch.no_grad(): print("base prob on new class before LoRA:", F.softmax(base64(Hd), 1)[:, 10].mean().item())
    for t in range(args.T):
        loss = F.cross_entropy(base64(Hd, W1=W0 + B @ A), y_priv); gA, gB = torch.autograd.grad(loss, [A, B])
        with torch.no_grad(): A -= args.lr * gA; B -= args.lr * gB
    print(f"LoRA loss after {args.T} steps: {loss.item():.3e}")
    A_T, B_T = A.detach(), B.detach()
    _, sB, VhB = torch.linalg.svd(B_T, full_matrices=False)
    print("singular values of B_T (first N+3):", [f"{v:.2e}" for v in sB[: args.N + 3].tolist()])
    gap = (sB[args.N - 1] / sB[args.N]).item(); print(f"excitation gap sigma_N/sigma_(N+1) = {gap:.2e}")
    Vq = VhB[: args.N].T; P_perp = torch.eye(args.r, dtype=torch.float64, device=dev) - Vq @ Vq.T; C = P_perp @ A_T
    s1 = (C @ Hd.T).norm().item() / (C.norm().item() * Hd.norm().item()); s2 = torch.linalg.matrix_rank(C, rtol=1e-10).item()
    s3 = (C - P_perp @ A0).norm().item() / C.norm().item()
    print(f"||C H|| / (||C|| ||H||) = {s1:.2e}   rank C = {s2} (expect {args.r - args.N})   ||C - P_perp A0|| / ||C|| = {s3:.2e}")
    return A_T, B_T, C, dict(gap=gap, CH_rel=s1, rank_C=s2, quotient=s3, loss=loss.item())

A0 = torch.randn(args.r, D, dtype=torch.float64, device=dev) / math.sqrt(D)          # same draw position as the raw run
print("=== raw-private LoRA (reproduces jobs 252897/252898) ===")
A_T_raw, B_T_raw, C_raw, san_raw = lora_and_certificate(H_raw.double().to(dev), A0)

# ---------------------------------------------------------------- chart (identical)
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
ae = AE(args.k).to(dev); opt = torch.optim.Adam(ae.parameters(), 1e-3)
for ep in range(args.ae_epochs):
    for i in torch.randperm(len(X_chart)).split(256):
        xb = X_chart[i]; opt.zero_grad(); F.mse_loss(ae(xb), xb).backward(); opt.step()
ae.eval()
for q_ in ae.parameters(): q_.requires_grad_(False)

# ---------------------------------------------------------------- the private set for THIS run
Hf_raw = H_raw.to(dev)
with torch.no_grad(): H_proj = ae(Hf_raw)                                                # chart projections of the raw privates
if args.private == "onchart":
    print("=== ON-CHART control: private inputs := G(enc(x_i)); same seed, base and chart ===")
    Hf = H_proj.detach()
    A_T, B_T, C, san = lora_and_certificate(Hf.double(), A0)
else:
    Hf = Hf_raw; A_T, B_T, C, san = A_T_raw, B_T_raw, C_raw, san_raw
Cn = C.float() / C.norm().float()

import kornia
def ssim(a, b):
    a = a.view(-1, 3, 32, 32).float(); b = b.view(-1, 3, 32, 32).float()
    return kornia.metrics.ssim(a, b, window_size=3).mean(dim=(1, 2, 3))

with torch.no_grad():
    floor_img = ae(Hf); floor_ssim = ssim(floor_img, Hf); ctrl = X_pub[: args.N].to(dev)
    res_floor = (Cn @ floor_img.T).norm(dim=0) / floor_img.norm(dim=1)
    res_truth = (Cn @ Hf.T).norm(dim=0) / Hf.norm(dim=1)
print("chart floor SSIM per private image:", [f"{v:.2f}" for v in floor_ssim.tolist()])
print("certificate residual at the private inputs:", [f"{v:.2e}" for v in res_truth.tolist()])
print("certificate residual at chart floor:", [f"{v:.2e}" for v in res_floor.tolist()])

# ---------------------------------------------------------------- certificate search (identical)
def residual(z):
    x = ae.G(z); return (Cn @ x.T).norm(dim=0) / x.norm(dim=1)
z = torch.randn(args.starts, args.k, device=dev, requires_grad=True); opt = torch.optim.Adam([z], 5e-2); t0 = time.time()
for it in range(3000):
    opt.zero_grad(); R = residual(z); (R ** 2).sum().backward(); opt.step()
    if it % 500 == 0: print(f"  it {it:4d}  median residual {R.median().item():.2e}  min {R.min().item():.2e}")
with torch.no_grad(): R = residual(z); X_found = ae.G(z)
print(f"search done in {time.time()-t0:.0f}s")

with torch.no_grad():
    S = torch.stack([ssim(X_found, Hf[i:i + 1].expand_as(X_found)) for i in range(args.N)], 1)
    best = S.max(0)
    err = torch.stack([(X_found - Hf[i:i + 1]).norm(dim=1) / Hf[i:i + 1].norm() for i in range(args.N)], 1)   # [starts, N] relative error
    nearest = err.argmin(1); err_near = err.min(1).values
    landed = err_near < 1e-2
    ctrl_ssim = torch.stack([ssim(X_found[best.indices[i]:best.indices[i] + 1], ctrl[i:i + 1]) for i in range(args.N)]).squeeze()
    raw_ssim = torch.stack([ssim(X_found[best.indices[i]:best.indices[i] + 1], Hf_raw[i:i + 1]) for i in range(args.N)]).squeeze()
    pd = torch.cdist(X_found, X_found); collapse = (pd < 1e-2 * X_found.norm(dim=1).mean()).float().mean().item()
print("\n=== RESULT ===")
print(f"private = {args.private};  starts with residual < 3x max chart-floor residual: {(R < 3 * res_floor.max()).sum().item()} / {args.starts}")
print(f"starts LANDED (relative error < 1e-2 vs a private input): {landed.sum().item()} / {args.starts};  per image: {[(landed & (nearest == i)).sum().item() for i in range(args.N)]}")
print(f"fraction of found-image pairs within 1% of each other (collapse): {collapse:.2f}")
print("min relative error per private input:", [f"{v:.2e}" for v in err.min(0).values.tolist()])
print("best SSIM found-vs-private-input:", [f"{v:.2f}" for v in best.values.tolist()])
print("best SSIM found-vs-RAW image    :", [f"{v:.2f}" for v in raw_ssim.tolist()])
print("chart floor SSIM                :", [f"{v:.2f}" for v in floor_ssim.tolist()])
print("same-class control SSIM         :", [f"{v:.2f}" for v in ctrl_ssim.tolist()])
print("residual of best landing        :", [f"{R[j].item():.2e}" for j in best.indices.tolist()])
print("residual at private inputs      :", [f"{v:.2e}" for v in res_truth.tolist()])

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [("private (raw)", Hf_raw), ("private input used for training" if args.private == "onchart" else "chart floor", Hf if args.private == "onchart" else floor_img),
        ("found (certificate only)", X_found[best.indices])]
fig, ax = plt.subplots(len(rows), args.N, figsize=(1.3 * args.N, 1.4 * len(rows)))
for i, (name, imgs) in enumerate(rows):
    for j in range(args.N):
        ax[i, j].imshow(imgs[j].view(3, 32, 32).permute(1, 2, 0).cpu().clamp(0, 1)); ax[i, j].axis("off")
    ax[i, 0].set_title(name, fontsize=7, loc="left")
fig.suptitle(f"CIFAR-100 class {args.cls100} as 11th class, private={args.private}, N={args.N}, r={args.r}, k={args.k}, T={args.T}", fontsize=8)
plt.tight_layout(); plt.savefig(f"{args.out}/cifar_certificate_{args.private}.png", dpi=150)
torch.save({"A_T": A_T.cpu(), "B_T": B_T.cpu(), "A0": A0.cpu(), "H_train": Hf.cpu(), "H_raw": H_raw, "priv_idx": priv_idx, "C": C.cpu(),
            "X_found": X_found.cpu(), "R": R.cpu(), "z": z.detach().cpu(), "res_floor": res_floor.cpu(), "res_truth": res_truth.cpu(),
            "sanity": san, "sanity_raw": san_raw, "private": args.private}, f"{args.out}/release_and_search.pt")
print(f"saved {args.out}/cifar_certificate_{args.private}.png and release_and_search.pt")
