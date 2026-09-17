#!/usr/bin/env python3
"""WP3 -- a pretrained image decoder as the chart: FIDELITY ONLY, no inversion, no release read.

Question (notes/plan_2026-09-18_cnn_ranklaw_newclass_charts.md, WP3): can the KL-VAE decoder of Stable Diffusion
(`stabilityai/sd-vae-ft-mse`), used as a LOCAL chart  G(w) = D(a + U w),  contain the private images to within the
landing gate at a width k below the identifiability cap (k <= 66 on the CIFAR releases)?  Every chart measured so far
was some PCA and PCA lost to the gate by 15x / 46x (e1b C7).  This script measures containment only; the pixel-PCA
baseline is computed in the SAME job on the SAME eight images in the C7 convention (two_walls.py).

Images -- the eight privates the releases are attacked on, selected EXACTLY as the oracle ladder selects them
(ladder_cell.py: torch.Generator().manual_seed(seed+7), randperm over the class's CIFAR-100 TEST split, first N)
and as the letters cell selects them (new_class.py --domain mnist: same generator, first randperm over the EMNIST
'a' TEST split).  Public pool = the class's TRAIN split, the pool the PCA charts are built from.

Measurements (relative pixel error ||x_hat - x|| / ||x|| in NATIVE space, 3x32x32 or 1x28x28, median + range over 8):
  0  resize round-trip floor       down(up(x))                       every arm below inherits it
  1  autoencoding ceiling          down(D(E(up(x))))  at factors x2 / x4 / x8        (E = posterior mean, no sampling)
  2  global latent PCA chart       decode of the top-k PCA projection of E(up(x)) in the PUBLIC pool's latent PCA
  3  local patch chart             min_w ||down(D(a + U w)) - x|| / ||x||, Adam in w, FP32, 2000 steps, 3 restarts
       anchors:  truth_nn      a = mean latent of the K public images nearest to E(x)      NOT attacker-available
                 truth_latent  a = E(x) itself (w = 0 IS the ceiling)                      NOT attacker-available
                 proxy_nn      a = mean latent of the K public images nearest to E(proxy), proxy = the pixel-PCA-k
                               projection of x = the ON-CHART RECOVERY, i.e. what the certificate attack returns
                               in a PCA chart when it lands                                attacker-available
       U = top-k PCA directions of those K latents (k_eff = min(k, K-1)); restart 0 starts at w = 0 (the anchor),
       restarts 1, 2 at N(0,1) in units of the neighbours' coordinate std.
  4  pixel PCA at the same k       C7 convention, plus the C7 index set (np.random.RandomState(seed)) as a cross-check

Resize path, identical in every decoder arm: up = exact pixel replication by an integer factor (x2 / x4 / x8 of the
native side: 64/128/256 for CIFAR, 56/112/224 for EMNIST), grey replicated to 3 channels, [0,1] -> [-1,1];  down = block
mean (area) back to the native size, channels averaged for grey, [-1,1] -> [0,1].  down(up(x)) == x exactly, so the
resize floor is zero and the ceiling is the decoder's alone (`--up bilinear` reproduces the 0.061 floor of job 355876).
Errors are reported UNCLAMPED (headline) and clamped to [0,1] (recorded).

  python -u -m experiments.decoder_chart.fidelity --image-sets mlp_motorcycle --ks 16 --Ks 64 --steps 200 --tag smoke
  python -u -m experiments.decoder_chart.fidelity --image-sets cnn_keyboard
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch
import torch.nn.functional as F

from experiments.cifar.cifar_newclass import load_cifar100_class
from experiments.exact_inversion.new_class import load_emnist_letters
from experiments.exact_inversion.lora_exact_inversion import git_hash

torch.set_default_dtype(torch.float32)   # the imported exact-inversion modules set FP64 at import time; this arm is FP32 by plan (job 355897 died on the mix)

DECODER = "stabilityai/sd-vae-ft-mse"
IMAGE_SETS = {   # gates are BRACKETS (audit 2026-09-18; ledger 0.1, A22): (last all-8 landing, first no-landing); compared at both ends
    "mlp_motorcycle": dict(domain="cifar", cls="motorcycle", data_root="data", shape=(3, 32, 32), gate=(0.0124, 0.0186),
                           gate_source="oracle ladder, MLP release: bracket 0.0124 (last all-8 landing) - 0.0186 (first failure)",
                           ckpt="models/exact_inversion/cifar10_mlp_newclass.pth"),
    "cnn_keyboard":   dict(domain="cifar", cls="keyboard", data_root="data", shape=(3, 32, 32), gate=(0.0045, 0.0090),
                           gate_source="oracle ladder, CNN release: bracket 0.0045 (last landing) - 0.0090 (first failure)",
                           ckpt="models/exact_inversion/cifar10_cnn_newclass.pth"),
    "mnist_letter_a": dict(domain="mnist", letter="a", data_root="dataset_reconstruction/data", shape=(1, 28, 28), gate=None,
                           gate_source="no MNIST gate measured yet (WP5); no shortfall is quoted",
                           ckpt="models/exact_inversion/mnist_mlp_strong.pth"),
}


def log(s): print(s, flush=True)


def rel_err(xh, x):
    """||x_hat - x|| / ||x|| per image, both (n, D) in native pixel space."""
    return torch.linalg.norm(xh - x, dim=1) / torch.linalg.norm(x, dim=1)


def stats(e, gate=None):
    e = e.detach().cpu()
    d = dict(err_median=float(e.median()), err_min=float(e.min()), err_max=float(e.max()), err_mean=float(e.mean()),
             err_per_image=[float(v) for v in e])
    if gate is not None:                       # the bracket: below its LOW end = lands as in the ladder; above its HIGH end = the ladder failed there
        d.update(below_gate_low=bool(d["err_median"] <= gate[0]), below_gate_high=bool(d["err_median"] <= gate[1]),
                 ratio_to_gate_low=d["err_median"] / gate[0], ratio_to_gate_high=d["err_median"] / gate[1],
                 n_images_below_gate_low=int((e <= gate[0]).sum()), n_images_below_gate_high=int((e <= gate[1]).sum()))
    return d


def ckpt_gate(path):
    """WP0: the frozen base's numbers as stored in the checkpoint. Informational only: WP3 reads NO release and the
    checkpoint's only role here is that its cells fixed which eight images are private (images-only, audit item 4)."""
    if not os.path.exists(path): return dict(ckpt=path, note="checkpoint not found")
    b = torch.load(path, map_location="cpu", weights_only=False)
    out = dict(ckpt=path, role="images-only: selects the private images; no release is read")
    for k in ("train_acc", "test_acc", "train_loss", "epochs"):
        if isinstance(b, dict) and k in b: out[k] = float(b[k])
    if "train_acc" not in out: out["note"] = "train accuracy / loss not recorded in this checkpoint"
    return out


def load_images(name, N, seed):
    """Public pool (train split) and the SAME eight privates as the ladder / letters cells. Returns float32 (n, D)."""
    ex = IMAGE_SETS[name]
    if ex["domain"] == "cifar":
        pool, cname = load_cifar100_class(ex["data_root"], ex["cls"])
        Pub, Pri = torch.tensor(pool["train"]), torch.tensor(pool["test"])
        g = torch.Generator().manual_seed(seed + 7); perm = torch.randperm(Pri.shape[0], generator=g)   # ladder_cell.py, verbatim
        idx = perm[:N].tolist()
        idx_c7 = np.random.RandomState(seed).permutation(len(Pri))[:N].tolist()                          # two_walls.py (C7)
    else:
        fl = load_emnist_letters(ex["data_root"], ex["letter"]); cname = f"emnist_letter_{ex['letter']}"
        Pub, Pri = torch.tensor(fl["train"][0], dtype=torch.float32), torch.tensor(fl["test"][0], dtype=torch.float32)
        g = torch.Generator().manual_seed(seed + 7); pf = torch.randperm(Pri.shape[0], generator=g)[:N]   # new_class.py, verbatim
        idx = pf.tolist(); idx_c7 = None
    return Pub.float(), Pri[idx].float().contiguous(), (Pri[idx_c7].float().contiguous() if idx_c7 else None), idx, idx_c7, cname


class Resizer:
    """The one resize path. up: native (n, D) -> (n, 3, s, s) in [-1, 1].  down: (n, 3, s, s) -> native (n, D) in [0, 1].
    s is an INTEGER multiple of the native side (x2 / x4 / x8: 64/128/256 for CIFAR, 56/112/224 for EMNIST), up is exact
    pixel replication and down is the block mean, so down(up(x)) == x and the floor of every arm is zero. (The first
    smoke run, job 355876, used bilinear up: its round-trip floor alone was 0.061 at x2, 5x the gate -- a resize
    artefact, not a decoder number; `--up bilinear` keeps that path available for comparison.)"""
    def __init__(self, shape, up="nearest"):
        self.C, self.H, self.W = shape; self.mode = up
        self.up_desc = ("nearest (exact pixel replication at integer factor)" if up == "nearest" else "bilinear, align_corners=False") + \
                       ", grey replicated to 3 channels, [0,1]->[-1,1]"
        self.down_desc = "area (block mean / adaptive average pooling) to native, channels averaged for grey, [-1,1]->[0,1]"

    def scale(self, f):
        assert self.H == self.W; return f * self.H

    def up(self, x, s):
        y = x.reshape(-1, self.C, self.H, self.W)
        if self.C == 1: y = y.expand(-1, 3, -1, -1)
        if self.mode == "nearest":
            assert s % self.H == 0, f"scale {s} is not an integer multiple of the native side {self.H}"
            y = F.interpolate(y, scale_factor=s // self.H, mode="nearest")
        else:
            y = F.interpolate(y, size=(s, s), mode="bilinear", align_corners=False)
        return y * 2 - 1

    def down(self, y):
        z = F.interpolate((y + 1) / 2, size=(self.H, self.W), mode="area")
        if self.C == 1: z = z.mean(1, keepdim=True)
        return z.reshape(z.shape[0], -1)


def load_vae(dev):
    from diffusers import AutoencoderKL
    import diffusers
    vae = AutoencoderKL.from_pretrained(DECODER, local_files_only=True, torch_dtype=torch.float32).to(dev).eval()   # fails loudly offline
    vae.requires_grad_(False)
    return vae, diffusers.__version__


@torch.no_grad()
def encode(vae, rs, x, s, bs=32):
    out = []
    for i in range(0, x.shape[0], bs):
        out.append(vae.encode(rs.up(x[i:i + bs], s)).latent_dist.mean)      # posterior mean, no sampling
    z = torch.cat(out, 0); return z.reshape(z.shape[0], -1)                   # flat (n, 4*(s/8)^2)


def decode_native(vae, rs, z, bs=8):
    """Flat latents (n, d) or (n, 4, h, h) -> native pixels (n, D)."""
    if z.dim() == 2: h = int(math.sqrt(z.shape[1] // 4)); z = z.reshape(-1, 4, h, h)
    return torch.cat([rs.down(vae.decode(z[i:i + bs]).sample) for i in range(0, z.shape[0], bs)], 0)


def pca(Z, k):
    """Top-k PCA of rows of Z: mean, directions (k, d) and per-direction coordinate std."""
    mu = Z.mean(0); _, S, Vh = torch.linalg.svd(Z - mu, full_matrices=False)
    k = min(k, Vh.shape[0]); return mu, Vh[:k], S[:k] / math.sqrt(max(Z.shape[0] - 1, 1))


def pixel_pca_proj(Pub, X, k):
    mu, V, _ = pca(Pub, k); Xc = X - mu
    return mu + (Xc @ V.T) @ V


def local_chart_fit(vae, rs, X, a, U, sd, steps, restarts, lr, chunk, gen, dev):
    """min_w ||down(D(a_i + U_i w)) - x_i|| / ||x_i|| by Adam, all images and restarts in one batch.
    a: (N, d) anchors, U: (N, k, d) directions, sd: (N, k) coordinate std (w is optimised in units of sd).
    Returns per-image best final error over restarts, the winning images, err at w = 0, and the trace."""
    N, k, d = U.shape; R = restarts
    v = torch.zeros(R, N, k, device=dev)
    if R > 1: v[1:] = torch.randn(R - 1, N, k, generator=gen).to(dev)              # restart 0 = the anchor itself
    v.requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    Xr = X.repeat(R, 1); a_r = a.repeat(R, 1); U_r = U.repeat(R, 1, 1); sd_r = sd.repeat(R, 1)
    lat_shape = (4, int(math.sqrt(d // 4)), int(math.sqrt(d // 4)))
    err0 = None; trace = []; e_traj_min = None
    for t in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        vf = v.reshape(R * N, k); errs = []
        for i in range(0, R * N, chunk):                                                   # graph built PER chunk (job 355900: a shared z was freed by the first backward)
            z = a_r[i:i + chunk] + torch.einsum("bk,bkd->bd", vf[i:i + chunk] * sd_r[i:i + chunk], U_r[i:i + chunk])
            xh = rs.down(vae.decode(z.reshape(-1, *lat_shape)).sample)
            e = rel_err(xh, Xr[i:i + chunk]); errs.append(e.detach())
            if t < steps: (e ** 2).sum().backward()
        e_all = torch.cat(errs).reshape(R, N)
        if t == 0: err0 = e_all[0].clone()                                             # restart 0 at w = 0: the anchor alone
        e_traj_min = e_all.min(0).values if e_traj_min is None else torch.minimum(e_traj_min, e_all.min(0).values)
        if t % max(1, steps // 10) == 0 or t == steps:
            trace.append((t, float(e_all.min(0).values.median()))); log(f"      step {t:5d}  median best-of-restarts err {trace[-1][1]:.4f}")
        if t < steps: opt.step(); sch.step()
    best = e_all.min(0)                                                                # FINAL error of the best restart (the reported number)
    with torch.no_grad():
        z = a_r + torch.einsum("bk,bkd->bd", v.reshape(R * N, k) * sd_r, U_r)
        Xh = decode_native(vae, rs, z.reshape(-1, *lat_shape), chunk).reshape(R, N, -1)
        Xbest = Xh[best.indices, torch.arange(N)]
    return best.values, Xbest, err0, trace, [int(i) for i in best.indices], e_traj_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-sets", nargs="+", default=list(IMAGE_SETS), choices=list(IMAGE_SETS))
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--factors", type=int, nargs="+", default=[2, 4, 8], help="decoder input side = factor x native side (64/128/256 CIFAR, 56/112/224 EMNIST), measurements 1 and 2")
    ap.add_argument("--local-factor", type=int, default=8, help="factor for the local chart (measurement 3)")
    ap.add_argument("--up", choices=["nearest", "bilinear"], default="nearest", help="up-resize; nearest = exact replication, zero floor")
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 32, 66, 128])
    ap.add_argument("--Ks", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--anchors", nargs="+", default=["proxy_nn", "truth_nn", "truth_latent"])
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--tag", default=""); ap.add_argument("--out-dir", default="results/decoder_chart")
    ap.add_argument("--set-seed-note", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fig-dir", default="figures/decoder_chart")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device)
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.fig_dir, exist_ok=True)
    jobid = os.environ.get("LSB_JOBID", "local"); tag = (a.tag + "_" if a.tag else "")
    out_path = os.path.join(a.out_dir, f"fidelity_{tag}{jobid}.jsonl")
    torch.manual_seed(a.seed); gen = torch.Generator().manual_seed(a.seed + 31)
    vae, dver = load_vae(dev)
    log(f"# decoder_chart fidelity  decoder={DECODER} diffusers={dver} torch={torch.__version__} fp32 device={dev} job={jobid} git={git_hash()}")
    log(f"# sets={a.image_sets} factors={a.factors} local_factor={a.local_factor} up={a.up} ks={a.ks} Ks={a.Ks} anchors={a.anchors} steps={a.steps} restarts={a.restarts} lr={a.lr}")
    common = dict(part="decoder_chart_fidelity", precision="fp32", decoder=DECODER, diffusers=dver, torch=torch.__version__,
                  N=a.N, seed=a.seed, git=git_hash(), host=socket.gethostname(), jobid=jobid, cmd=" ".join(sys.argv))

    def emit(row):
        row = dict(**common, **row)
        with open(out_path, "a") as f: f.write(json.dumps(row) + "\n")
        return row

    for name in a.image_sets:
        ex = IMAGE_SETS[name]; rs = Resizer(ex["shape"], a.up); t_set = time.time()
        scales = [rs.scale(f) for f in a.factors]; local_scale = rs.scale(a.local_factor)
        Pub, X, X_c7, idx, idx_c7, cname = load_images(name, a.N, a.seed)
        Pub, X = Pub.to(dev), X.to(dev)
        base = ckpt_gate(ex["ckpt"])
        setinfo = dict(image_set=name, domain=ex["domain"], class_name=cname, native_shape=list(ex["shape"]), private_indices=idx,
                       selection="ladder_cell.py / new_class.py: torch.Generator().manual_seed(seed+7), randperm over the class TEST split",
                       n_public=int(Pub.shape[0]), public_pool="class TRAIN split (the PCA chart's pool)",
                       gate=ex["gate"], gate_source=ex["gate_source"], base_model=base, release_read=False,
                       resize_up=rs.up_desc, resize_down=rs.down_desc)
        log(f"\n=== {name} ('{cname}')  privates idx={idx}  public pool {tuple(Pub.shape)}  gate={ex['gate']}  base={base}")
        keep = dict(x_truth=X.cpu())

        def report(label, e, gate=ex["gate"]):
            st = stats(e, gate)
            if gate is None: flag = ""
            elif st["below_gate_low"]: flag = "  BELOW the gate bracket's low end"
            elif st["below_gate_high"]: flag = "  INSIDE the gate bracket"
            else: flag = f"  ({st['ratio_to_gate_low']:.1f}x low end, {st['ratio_to_gate_high']:.1f}x high end)"
            log(f"  {label:<58s} median {st['err_median']:.4f}  range {st['err_min']:.4f}-{st['err_max']:.4f}{flag}")
            return st

        # ---- 0 + 1: resize floor and the autoencoding ceiling, scale by scale
        Zpub = {}; Zx = {}; ceil_err = {}
        for s in scales:
            with torch.no_grad():
                e0 = rel_err(rs.down(rs.up(X, s)), X)
                st0 = report(f"[0] resize round-trip floor        scale {s}", e0)
                emit(dict(**setinfo, measurement="resize_floor", scale=s, attacker_available=True, **st0))
                Zx[s] = encode(vae, rs, X, s); Xae = decode_native(vae, rs, Zx[s], a.chunk)
                e1 = rel_err(Xae, X); e1c = rel_err(Xae.clamp(0, 1), X); ceil_err[s] = e1
                st1 = report(f"[1] autoencoding ceiling D(E(x))   scale {s}", e1)
                emit(dict(**setinfo, measurement="ae_ceiling", scale=s, latent_dim=int(Zx[s].shape[1]), attacker_available=True,
                          err_clamped_median=float(e1c.median()), **st1))
                keep[f"x_ae_{s}"] = Xae.cpu()
                t0 = time.time(); Zpub[s] = encode(vae, rs, Pub, s); log(f"      encoded public pool at {s}: {tuple(Zpub[s].shape)} in {time.time()-t0:.0f}s")

        # ---- 4: pixel PCA at the same k (C7 convention), first so the proxy exists for measurement 3
        proxies = {}
        for k in a.ks:
            with torch.no_grad():
                Xp = pixel_pca_proj(Pub, X, k); proxies[k] = Xp
                st4 = report(f"[4] pixel PCA chart                k={k}", rel_err(Xp, X))
                row = dict(**setinfo, measurement="pixel_pca", k=k, attacker_available=True,
                           note="on-chart recovery: the certificate attack in this chart returns the projection of the truth", **st4)
                if X_c7 is not None:
                    e_c7 = rel_err(pixel_pca_proj(Pub, X_c7.to(dev), k), X_c7.to(dev))
                    row.update(c7_indices=idx_c7, c7_err_mean=float(e_c7.mean()), c7_err_median=float(e_c7.median()),
                               c7_note="same computation on the two_walls.py (np.random.RandomState(seed)) index set, for comparison with C7")
                    log(f"      (C7 index set {idx_c7}: mean {float(e_c7.mean()):.4f})")
                emit(row); keep[f"x_pixpca_k{k}"] = Xp.cpu()

        # ---- 2: global latent PCA chart of the public pool, decoded
        for s in scales:
            for k in a.ks:
                with torch.no_grad():
                    mu, V, _ = pca(Zpub[s], k); zh = mu + ((Zx[s] - mu) @ V.T) @ V
                    Xg = decode_native(vae, rs, zh, a.chunk); e2 = rel_err(Xg, X); e2c = rel_err(Xg.clamp(0, 1), X)
                    st2 = report(f"[2] global latent PCA chart        scale {s} k={k}", e2)
                    emit(dict(**setinfo, measurement="global_latent_pca", scale=s, k=k, k_eff=int(V.shape[0]), latent_dim=int(V.shape[1]),
                              attacker_available=True, err_clamped_median=float(e2c.median()),
                              note="projection of E(x) onto the public latent PCA, decoded (on-chart recovery convention)", **st2))
                    keep[f"x_glob_s{s}_k{k}"] = Xg.cpu()

        # ---- 3: local patch chart, Adam in w
        s = local_scale; Zp, Ztruth = Zpub[s], Zx[s]; best_attacker = None
        for K in a.Ks:
            K_ = min(K, Zp.shape[0])
            for k in a.ks:
                for anchor in a.anchors:
                    with torch.no_grad():
                        q = encode(vae, rs, proxies[k], s) if anchor == "proxy_nn" else Ztruth       # the query latent
                        nn_idx = torch.cdist(q, Zp).topk(K_, largest=False).indices                    # (N, K) neighbours in latent space
                        A_, U_, S_ = [], [], []
                        for i in range(a.N):
                            mu_i, V_i, sd_i = pca(Zp[nn_idx[i]], k)
                            A_.append(Ztruth[i] if anchor == "truth_latent" else mu_i); U_.append(V_i); S_.append(sd_i)
                        a_i, U_i, sd_i = torch.stack(A_), torch.stack(U_), torch.stack(S_); k_eff = int(U_i.shape[1])
                    t0 = time.time()
                    log(f"  [3] local chart  scale {s} K={K_} k={k} (k_eff={k_eff}) anchor={anchor}  Adam {a.steps} steps x {a.restarts} restarts")
                    e3, Xl, e_w0, trace, win, e_traj = local_chart_fit(vae, rs, X, a_i, U_i, sd_i, a.steps, a.restarts, a.lr, a.chunk, gen, dev)
                    e3c = rel_err(Xl.clamp(0, 1), X)
                    st3 = report(f"[3] local chart K={K_} k={k} anchor={anchor}", e3)
                    avail = anchor == "proxy_nn"
                    # solver check (audit item 3): the fit is solver-bounded, pixel PCA is closed-form. Restart 0 starts at w = 0, so
                    # the best-of-restarts FINAL error may never exceed the w = 0 error; at the oracle anchor a = E(x), w = 0 IS the
                    # autoencoding ceiling, and a fit ending above the ceiling is a solver failure, not a chart number.
                    solver = dict(err_w0_median=float(e_w0.median()), err_w0_per_image=[float(v) for v in e_w0],
                                  err_trajmin_median=float(e_traj.median()), ended_above_w0=bool((e3 > e_w0 + 1e-6).any()))
                    if anchor == "truth_latent":
                        ceil = ceil_err[s]; dev_ = float((e_w0 - ceil).abs().max())
                        solver.update(ceiling_median=float(ceil.median()), w0_minus_ceiling_maxabs=dev_, w0_reproduces_ceiling=bool(dev_ <= 1e-5),
                                      solver_failure=bool((e3 > ceil + 1e-6).any()))
                        log(f"      solver check: w=0 {solver['err_w0_median']:.5f} | ceiling {solver['ceiling_median']:.5f} (max |diff| {dev_:.1e}) "
                            f"| best-of-restarts {st3['err_median']:.5f}  -> {'SOLVER FAILURE' if solver['solver_failure'] else 'ok'}")
                    else:
                        log(f"      w=0 (anchor alone) {solver['err_w0_median']:.4f} | best-of-restarts {st3['err_median']:.4f}"
                            f"{'  SOLVER-LIMITED (ended above w=0)' if solver['ended_above_w0'] else ''}")
                    row = emit(dict(**setinfo, measurement="local_chart", scale=s, K=K_, k=k, k_eff=k_eff, anchor=anchor, attacker_available=avail,
                                    steps=a.steps, restarts=a.restarts, lr=a.lr, lr_schedule="cosine to 1% of lr", optimizer="Adam",
                                    err_clamped_median=float(e3c.median()), winning_restart=win, trace=trace, seconds=time.time() - t0,
                                    note=("neighbours of the pixel-PCA-k on-chart recovery (attacker-available)" if avail else
                                          "NOT attacker-available: anchor/neighbours use the private image itself"), **solver, **st3))
                    keep[f"x_local_K{K_}_k{k}_{anchor}"] = Xl.cpu()
                    if avail and (best_attacker is None or row["err_median"] < best_attacker[0]): best_attacker = (row["err_median"], K_, k, Xl.cpu())

        torch.save(dict(**keep, indices=idx, image_set=name, args=vars(a)), os.path.join(a.out_dir, f"fidelity_{tag}{name}_{jobid}.pth"))
        # ---- grid: truth / D(E(x)) / best attacker-available local chart / pixel PCA at k=66 (or the largest k run)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        k_show = 66 if 66 in a.ks else max(a.ks)
        rows_ = [("truth", keep["x_truth"]), (f"D(E(x)) scale {s}", keep[f"x_ae_{s}"])]
        if best_attacker: rows_.append((f"best local chart K={best_attacker[1]} k={best_attacker[2]} (proxy_nn) med {best_attacker[0]:.3f}", best_attacker[3]))
        rows_.append((f"pixel PCA k={k_show}", keep[f"x_pixpca_k{k_show}"]))
        fig, axes = plt.subplots(len(rows_), a.N, figsize=(1.3 * a.N, 1.45 * len(rows_)))
        C, H, W = ex["shape"]
        for r_, (lab, T) in enumerate(rows_):
            for c in range(a.N):
                ax = axes[r_, c]; ax.axis("off"); im = T[c].reshape(C, H, W).clamp(0, 1)
                ax.imshow(im.permute(1, 2, 0).numpy() if C == 3 else im[0].numpy(), cmap=None if C == 3 else "gray", vmin=0, vmax=1)
            axes[r_, 0].set_title(lab, fontsize=7, loc="left")
        fig.suptitle(f"{name} ('{cname}') -- decoder chart fidelity, job {jobid}", fontsize=8); plt.tight_layout()
        fp = os.path.join(a.fig_dir, f"fidelity_{tag}{name}_{jobid}.png"); plt.savefig(fp, dpi=130); plt.close(fig)
        log(f"# figure {fp}   ({time.time()-t_set:.0f}s for this image set)")
    log(f"# rows -> {out_path}")


if __name__ == "__main__":
    main()
