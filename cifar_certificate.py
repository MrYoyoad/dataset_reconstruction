"""
CIFAR-10 replica of the EMNIST certificate experiment (Figure 2 analogue).

Setting
  base    : MLP on pixels, 3072 -> 512 -> 256 -> 11 logits, pretrained on CIFAR-10 (class 10 never seen)
  private : N images of one CIFAR-100 class, used as the 11th class
  LoRA    : first layer only (input = pixels, so one input per image and q = N), B0 = 0, A0 Gaussian
  train   : full-batch vanilla SGD, float64, T steps
  release : (A_T, B_T) raw factors
  attack  : C = P_{row(B_T)^perp} A_T ; search  min_z ||C G(z)||^2 / (||C||^2 ||G(z)||^2)
            over a NONLINEAR decoder chart G trained on public CIFAR-100 images (private ones excluded),
            from random starts.  A linear PCA chart cannot separate the N images with the certificate
            alone (the zero set is then a single point or one affine subspace) -- see HOOK below.
  report  : certificate check (||CH||, rank C), landings per private image, SSIM found-vs-true,
            chart floor SSIM (decoder(encoder(x_i)) vs x_i), same-class control SSIM.

Run:  python cifar_certificate.py --N 8 --r 64 --k 32 --T 200 --cls100 0 --starts 400
"""
import argparse, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as tf

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=8)           # private images
p.add_argument("--r", type=int, default=64)          # LoRA rank
p.add_argument("--k", type=int, default=32)          # chart (decoder latent) dimension
p.add_argument("--T", type=int, default=200)         # LoRA SGD steps
p.add_argument("--lr", type=float, default=0.05)     # LoRA SGD step size
p.add_argument("--cls100", type=int, default=0)      # CIFAR-100 class used as the 11th class (0 = apple)
p.add_argument("--starts", type=int, default=400)    # random starts for the certificate search
p.add_argument("--pre_epochs", type=int, default=15)
p.add_argument("--ae_epochs", type=int, default=40)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", type=str, default="cifar_cert_out")
args = p.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out, exist_ok=True)
D = 3 * 32 * 32

# ---------------------------------------------------------------- data
T10 = tf.ToTensor()
c10_tr = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=T10)
c10_te = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=T10)
c100_tr = torchvision.datasets.CIFAR100("data", train=True, download=True, transform=T10)

def stack(ds, idx):
    return torch.stack([ds[i][0] for i in idx]).reshape(len(idx), -1)  # [n, 3072] in [0,1]

# CIFAR-100 images of the chosen class: N private, the rest public (for the chart and the control)
idx100 = [i for i, y in enumerate(c100_tr.targets) if y == args.cls100]
perm = np.random.permutation(len(idx100))
priv_idx = [idx100[i] for i in perm[:args.N]]
pub_idx  = [idx100[i] for i in perm[args.N:]]
H_priv = stack(c100_tr, priv_idx)                    # [N, 3072]  the private images (rows)
X_pub  = stack(c100_tr, pub_idx)                     # public same-class images
print(f"private {H_priv.shape}, public same-class {X_pub.shape}")

# ---------------------------------------------------------------- base model (fp32 pretrain)
class MLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.l1 = nn.Linear(D, 512); s.l2 = nn.Linear(512, 256); s.l3 = nn.Linear(256, 11)
    def forward(s, x, W1=None):
        h = F.gelu(F.linear(x, s.l1.weight if W1 is None else W1, s.l1.bias))
        h = F.gelu(s.l2(h)); return s.l3(h)

base = MLP().to(dev)
opt = torch.optim.Adam(base.parameters(), 1e-3)
ld = torch.utils.data.DataLoader(c10_tr, batch_size=256, shuffle=True, num_workers=2)
for ep in range(args.pre_epochs):
    for x, y in ld:
        x = x.reshape(len(x), -1).to(dev); y = y.to(dev)
        opt.zero_grad(); F.cross_entropy(base(x), y).backward(); opt.step()
with torch.no_grad():
    xt = torch.stack([c10_te[i][0] for i in range(2000)]).reshape(2000, -1).to(dev)
    yt = torch.tensor(c10_te.targets[:2000]).to(dev)
    acc = (base(xt).argmax(1) == yt).float().mean().item()
print(f"base CIFAR-10 acc (2k test) = {acc:.3f}")

# ---------------------------------------------------------------- LoRA on first layer, float64, vanilla SGD
base64 = MLP().to(dev).double(); base64.load_state_dict(base.state_dict())
for q_ in base64.parameters(): q_.requires_grad_(False)
W0 = base64.l1.weight.detach()                       # [512, 3072]
Hd = H_priv.double().to(dev)                         # [N, 3072]
y_priv = torch.full((args.N,), 10, device=dev)       # 11th class
A0 = torch.randn(args.r, D, dtype=torch.float64, device=dev) / math.sqrt(D)
A = A0.clone().requires_grad_(True)
B = torch.zeros(512, args.r, dtype=torch.float64, device=dev, requires_grad=True)
with torch.no_grad():
    print("base prob on new class before LoRA:", F.softmax(base64(Hd), 1)[:, 10].mean().item())
for t in range(args.T):
    loss = F.cross_entropy(base64(Hd, W1=W0 + B @ A), y_priv)
    gA, gB = torch.autograd.grad(loss, [A, B])
    with torch.no_grad():
        A -= args.lr * gA; B -= args.lr * gB
print(f"LoRA loss after {args.T} steps: {loss.item():.3e}")
A_T, B_T = A.detach(), B.detach()

# ---------------------------------------------------------------- certificate from the release alone
_, sB, VhB = torch.linalg.svd(B_T, full_matrices=False)   # rows of B_T live in R^r ; right singular vectors
print("singular values of B_T (first N+3):", [f"{v:.2e}" for v in sB[: args.N + 3].tolist()])
gap = (sB[args.N - 1] / sB[args.N]).item() if len(sB) > args.N else float("inf")
print(f"excitation gap sigma_N/sigma_(N+1) = {gap:.2e}   (attacker would pick q from this gap)")
q = args.N                                                  # self-audit: we know N; attacker uses the gap
Vq = VhB[:q].T                                              # [r, q] orthonormal basis of S = row(B_T)
P_perp = torch.eye(args.r, dtype=torch.float64, device=dev) - Vq @ Vq.T
C = P_perp @ A_T                                            # [r, 3072]
CH = C @ Hd.T
print(f"||C H|| / (||C|| ||H||) = {CH.norm().item() / (C.norm().item() * Hd.norm().item()):.2e}   (should be ~1e-15)")
print(f"rank C = {torch.linalg.matrix_rank(C, rtol=1e-10).item()}   (expect r - N = {args.r - args.N})")
Cq = P_perp @ A0
print(f"||C - P_perp A0|| / ||C|| = {(C - Cq).norm().item() / C.norm().item():.2e}   (quotient form, expect ~1e-15)")
Cn = C.float() / C.norm().float()                           # normalized C for the search (fp32 ok here)

# ---------------------------------------------------------------- chart: small conv autoencoder on PUBLIC images
# Public = same-class CIFAR-100 images minus the private ones, plus all CIFAR-10 train images (more variety).
X_chart = torch.cat([X_pub, stack(c10_tr, list(range(0, 50000, 5)))]).to(dev)   # ~10.5k images
class AE(nn.Module):
    def __init__(s, k):
        super().__init__()
        s.enc = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
                              nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Flatten(), nn.Linear(128 * 16, k))
        s.dec_fc = nn.Linear(k, 128 * 16)
        s.dec = nn.Sequential(nn.GELU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
                              nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())
    def G(s, z):  # decoder = the chart, R^k -> [0,1]^3072
        return s.dec(s.dec_fc(z).view(-1, 128, 4, 4)).reshape(len(z), -1)
    def forward(s, x): return s.G(s.enc(x.view(-1, 3, 32, 32)))
ae = AE(args.k).to(dev); opt = torch.optim.Adam(ae.parameters(), 1e-3)
for ep in range(args.ae_epochs):
    for i in torch.randperm(len(X_chart)).split(256):
        xb = X_chart[i]; opt.zero_grad(); F.mse_loss(ae(xb), xb).backward(); opt.step()
ae.eval()
for q_ in ae.parameters(): q_.requires_grad_(False)

# ---------------------------------------------------------------- SSIM (kornia window 3 on [0,1], as in the thesis)
try:
    import kornia
    def ssim(a, b):
        a = a.view(-1, 3, 32, 32).float(); b = b.view(-1, 3, 32, 32).float()
        return kornia.metrics.ssim(a, b, window_size=3).mean(dim=(1, 2, 3))
except ImportError:
    def ssim(a, b):  # fallback: 3x3 uniform window
        a = a.view(-1, 3, 32, 32).float(); b = b.view(-1, 3, 32, 32).float()
        mu_a = F.avg_pool2d(a, 3, 1); mu_b = F.avg_pool2d(b, 3, 1)
        va = F.avg_pool2d(a * a, 3, 1) - mu_a ** 2; vb = F.avg_pool2d(b * b, 3, 1) - mu_b ** 2
        cov = F.avg_pool2d(a * b, 3, 1) - mu_a * mu_b
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
        return s.mean(dim=(1, 2, 3))

Hf = H_priv.to(dev)
with torch.no_grad():
    floor_img = ae(Hf)                                       # chart floor: what the chart can represent
    floor_ssim = ssim(floor_img, Hf)
    ctrl = X_pub[:args.N].to(dev)                            # same-class control images (not private)
print("chart floor SSIM per private image:", [f"{v:.2f}" for v in floor_ssim.tolist()])
with torch.no_grad():
    res_floor = (Cn @ floor_img.T).norm(dim=0) / floor_img.norm(dim=1)
print("certificate residual at chart floor:", [f"{v:.2e}" for v in res_floor.tolist()])

# ---------------------------------------------------------------- certificate search from random starts
def residual(z):                                             # ||C G(z)|| / (||C|| ||G(z)||), per start
    x = ae.G(z); return (Cn @ x.T).norm(dim=0) / x.norm(dim=1)

z = torch.randn(args.starts, args.k, device=dev, requires_grad=True)
opt = torch.optim.Adam([z], 5e-2)
t0 = time.time()
for it in range(3000):
    opt.zero_grad(); R = residual(z); (R ** 2).sum().backward(); opt.step()
    if it % 500 == 0: print(f"  it {it:4d}  median residual {R.median().item():.2e}  min {R.min().item():.2e}")
with torch.no_grad():
    R = residual(z); X_found = ae.G(z)
print(f"search done in {time.time()-t0:.0f}s")

# match landings to private images by SSIM; a landing counts if its residual is near the chart floor
with torch.no_grad():
    S = torch.stack([ssim(X_found, Hf[i:i + 1].expand_as(X_found)) for i in range(args.N)], 1)  # [starts, N]
    best_per_img = S.max(0)
    ctrl_ssim = torch.stack([ssim(X_found[best_per_img.indices[i]:best_per_img.indices[i]+1], ctrl[i:i+1]) for i in range(args.N)]).squeeze()
    landed = (R < 3 * res_floor.max())
print("\n=== RESULT ===")
print(f"starts that reached the certificate floor: {landed.sum().item()} / {args.starts}")
print("best SSIM found-vs-true  :", [f"{v:.2f}" for v in best_per_img.values.tolist()])
print("chart floor SSIM         :", [f"{v:.2f}" for v in floor_ssim.tolist()])
print("same-class control SSIM  :", [f"{v:.2f}" for v in ctrl_ssim.tolist()])
print("residual of best landing :", [f"{R[j].item():.2e}" for j in best_per_img.indices.tolist()])

# ---------------------------------------------------------------- figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [("private (raw)", Hf), ("chart floor", floor_img), ("found (certificate only)", X_found[best_per_img.indices])]
fig, ax = plt.subplots(len(rows), args.N, figsize=(1.3 * args.N, 1.4 * len(rows)))
for i, (name, imgs) in enumerate(rows):
    for j in range(args.N):
        ax[i, j].imshow(imgs[j].view(3, 32, 32).permute(1, 2, 0).cpu().clamp(0, 1)); ax[i, j].axis("off")
    ax[i, 0].set_title(name, fontsize=7, loc="left")
fig.suptitle(f"CIFAR-100 class {args.cls100} as 11th class, N={args.N}, r={args.r}, k={args.k}, T={args.T}", fontsize=8)
plt.tight_layout(); plt.savefig(f"{args.out}/cifar_certificate.png", dpi=150)
torch.save({"A_T": A_T.cpu(), "B_T": B_T.cpu(), "A0": A0.cpu(), "H": H_priv, "priv_idx": priv_idx, "C": C.cpu()}, f"{args.out}/release.pt")
print(f"saved {args.out}/cifar_certificate.png and release.pt")

# ---------------------------------------------------------------- HOOK: Figure-3 analogue
# For the PCA / exact-channel variant, load release.pt into lora_exact_inversion.py and use a PCA chart
# fitted on X_pub (k up to m+r-N-1 = 11+r-N-1). The certificate alone cannot separate the N images on a
# linear chart; that experiment needs the replay solver.
