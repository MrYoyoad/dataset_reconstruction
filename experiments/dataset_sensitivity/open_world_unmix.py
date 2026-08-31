"""Open-world unmixing of the exact span (A₀=0 branch) — the thesis pixel-reconstruction milestone, cleanest form.

For A₀=0 first-layer LoRA at N≤r we hold row(ΔW)=span{x0_i} EXACTLY, noise-free. Drop the gallery: recover the N
private images from the span alone = "find N images inside a known N-dim subspace" = the noise-free superposition
problem. Method 1 = FastICA (Cocktail-Party): the span basis columns are linear mixtures of the N independent
image sources; unmix (numpy FastICA — rec env has no sklearn). Sign/scale ambiguity resolved WITHOUT truth
(nonnegativity + [0,1] pixel fit). Eval hygiene: Hungarian-match to the private images; report direction cosine
(sign-invariant, oracle-free), reconstruction SSIM, mean-image baseline SSIM, margin; N sweep 2→r. Ceilings:
closed-world selector = 1.0 (exact), mean-image = floor. Pre-register: recognizable = SSIM>baseline AND margin>0
on ≥N−1 of N. SCOPE: A₀=0 first-layer, N≤r, open-world, this-attacker. bsub GPU (adapter) + CPU (ICA).
"""
import argparse, os, math, torch, numpy as np
from scipy.optimize import linear_sum_assignment
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set
from experiments.data_utils import _load_dataset, _get_binary_label

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/open_world_unmix"
NS = [2, 3, 4, 6, 8]
RANK, T, LR, ACT = 8, 200, 0.5, "gelu"


def priv_images(ds, N, seed):
    tgt = ds.targets if torch.is_tensor(ds.targets) else torch.tensor(ds.targets)
    g = torch.Generator().manual_seed(seed); imgs = []
    per = {0: N - N // 2, 1: N // 2}
    for d in (0, 1):
        idx = (tgt == d).nonzero(as_tuple=True)[0]
        for i in idx[torch.randperm(len(idx), generator=g)[:per[d]]]:
            imgs.append((ds.data[int(i)].to(torch.float64).view(-1) / 255.0, float(_get_binary_label(int(d)))))
    x = torch.stack([a for a, _ in imgs]); y = torch.tensor([b for _, b in imgs])
    return x, y


def fastica(O, n_comp, iters=300, seed=0):
    """O: (n_samples=784, n_channels=n_comp). Returns sources S (784, n_comp), each column an image direction."""
    rng = np.random.default_rng(seed)
    Oc = O - O.mean(0, keepdims=True)
    cov = Oc.T @ Oc / Oc.shape[0]
    d, E = np.linalg.eigh(cov); d = np.maximum(d, 1e-12)
    Wwhite = E @ np.diag(1.0 / np.sqrt(d)) @ E.T
    Z = Oc @ Wwhite                                            # (784, n_comp) whitened
    W = np.zeros((n_comp, n_comp))
    for k in range(n_comp):
        w = rng.standard_normal(n_comp)
        for _ in range(iters):
            wx = Z @ w
            g = np.tanh(wx); gp = 1 - g ** 2
            w_new = (Z * g[:, None]).mean(0) - gp.mean() * w
            for j in range(k):                                # deflation orthogonalization
                w_new -= (w_new @ W[j]) * W[j]
            w_new /= np.linalg.norm(w_new) + 1e-12
            if abs(abs(w_new @ w) - 1) < 1e-8:
                w = w_new; break
            w = w_new
        W[k] = w
    return Z @ W.T                                            # (784, n_comp) sources


def ssim(a, b):
    a, b = a.ravel(), b.ravel(); mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var(); cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def reconstruct(src, ds_mean):
    """Resolve ICA sign/scale WITHOUT truth: pick sign + scale minimizing out-of-[0,1] mass after +mean."""
    best = None
    for sign in (1.0, -1.0):
        s = sign * src / (src.std() + 1e-12)
        for beta in (0.15, 0.25, 0.35, 0.5):
            img = np.clip(ds_mean + beta * s, 0, 1)
            clip_loss = np.abs(ds_mean + beta * s - img).mean()
            if best is None or clip_loss < best[0]:
                best = (clip_loss, img)
    return best[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ds = _load_dataset("mnist", train=True); act = make_activation(ACT)
    xr, yr, _ = build_set(2, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, ds_mean = _honest_target(xr, yr, T, RANK, ACT, LR, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]; dsm = ds_mean.reshape(-1).cpu().numpy()
    B0_atk = draw_B0(900, out_f, RANK, dev)
    print(f"[unmix] A₀=0 first-layer LoRA, r={RANK} | N-sweep {NS} | FastICA on the exact span")

    summary = {}
    for N in NS:
        cos_all, ssim_all, base_all, results = [], [], [], []
        for trial in range(6):
            x, y = priv_images(ds, N, seed=100 * N + trial)
            x = x.to(dev); x0 = x - ds_mean
            _, _, _, dWt = train_adapter(frozen, b0, B0_atk, x0, y.to(dev), LR, T, act, RANK)
            svd = torch.linalg.svd(dWt.detach().to("cpu", torch.float64), full_matrices=False)
            keep = int((svd.S > 1e-6 * svd.S[0]).sum())
            V = svd.Vh[:keep].transpose(-1, -2).cpu().numpy()             # (784, keep) span basis
            S = fastica(V, keep, seed=trial)                             # (784, keep) recovered sources
            x0_np = x0.cpu().numpy(); x_np = x.cpu().numpy()
            # direction cosine (sign-invariant) between each source and each true centered image
            Sn = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-12)
            X0n = x0_np.T / (np.linalg.norm(x0_np, axis=1, keepdims=True).T + 1e-12)
            cosM = np.abs(Sn.T @ X0n)                                     # (keep, N)
            if keep < N:                                                 # N>r: fewer sources than images (superposition)
                cosM = np.pad(cosM, ((0, N - keep), (0, 0)))
            ri, ci = linear_sum_assignment(-cosM)
            recon = [reconstruct(S[:, i] if i < keep else np.zeros(784), dsm) for i in ri]
            for r_i, c_i in zip(range(len(ci)), ci):
                cos_all.append(cosM[ri[r_i], c_i]); ssim_all.append(ssim(recon[r_i], x_np[c_i]))
                base_all.append(ssim(dsm, x_np[c_i]))
        cos_all, ssim_all, base_all = map(np.array, (cos_all, ssim_all, base_all))
        margin = ssim_all - base_all
        frac_recog = float((margin > 0).mean())
        summary[N] = dict(cos=float(cos_all.mean()), ssim=float(ssim_all.mean()), base=float(base_all.mean()),
                          margin=float(margin.mean()), frac_recog=frac_recog)
        tag = "N≤r" if N <= RANK else "N>r"
        print(f"  N={N} ({tag}): dir-cosine={cos_all.mean():.3f} | recon SSIM={ssim_all.mean():.3f} vs "
              f"mean-baseline {base_all.mean():.3f} (margin {margin.mean():+.3f}) | frac SSIM>baseline={frac_recog:.2f}")

    print(f"\n  [ceilings: closed-world selector = 1.0 (exact) · mean-image = the baseline shown]")
    print(f"  [SCOPE: A₀=0 first-layer, N≤r, open-world (no gallery), this-attacker; recognizable = SSIM>baseline & margin>0]")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(summary=summary, NS=NS, rank=RANK), os.path.join(RESULTS, "unmix.pth"))
        print(f"[saved] {RESULTS}/unmix.pth")


if __name__ == "__main__":
    main()
