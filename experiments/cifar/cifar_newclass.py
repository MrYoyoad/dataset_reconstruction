#!/usr/bin/env python3
"""A FULLY TRAINED CIFAR-10 backbone (MLP *or* CNN), a weird 11th class added by head LoRA, and the certificate attack.

Why this cell exists. The replica's base (`cifar_certificate.py`) is a 15-epoch MLP at 54.4%, so "was the base
actually trained?" was open. Here the frozen backbone is trained to convergence and BOTH gates are reported: test
accuracy (does it generalise) and train accuracy (has it converged -- the sense that matters for "fully trained",
since a pixel MLP's test ceiling on CIFAR-10 is ~55% however long you train it).

  --arch mlp   THE PROJECT'S OWN STRUCTURE (Haim et al.): 3072 - 1000 - 1000 - 10, GELU, bias on the first layer
               only, exactly `train_cifar_backbone.py` / `TrainedBackbone`. The certificate acts on the 1000-dim
               penultimate features. An MLP is expected to work and this arm is the check that it does.
  --arch cnn   a conv stack trained to >= 80%, penultimate 256-dim -- the deployment-shaped arm (frozen trained
               encoder + LoRA head), included so the result is not an artefact of a weak pixel model.
  --overtrain  THE ORIGINAL PAPER'S REGIME (Haim et al.): no augmentation, no weight decay, train past zero error
               until the loss collapses -- an interpolating, large-margin model. This is the adversarial case for
               the attack's mechanism, because such a model is confident on everything it has already seen (Step 21:
               a 98% model records almost nothing of confident examples) -- so if the certificate still hands back
               the new class here, the recording is driven by the new class being new, not by a weak base model.
               Reports the train loss and the margin distribution, which is what "over-trained" means numerically.

Setting (the arm that lands, from the layer study of 2026-09-06): LoRA on the HEAD (input = the CNN's penultimate
features, so the certificate sits behind the whole conv stack and blends of the private features are not features of
any chart image), 11th head row zeroed, r = 64, B0 = 0, full-batch vanilla SGD in float64, chart = PCA of PUBLIC
images of the new class, private images from a held-out split, certificate C = P_{row(B_T)^perp} A_T, search =
Levenberg-Marquardt on ||C phi(G(z))|| / ||A_T phi(G(z))|| from random starts at the public coordinate scale.

New classes (--newclass), chosen to be visually alien to CIFAR-10:
    cifar100:<name|idx>   any CIFAR-100 fine class -- e.g. keyboard, skyscraper, mushroom, clock, lobster
    fashion:<name>        a FashionMNIST class rendered at 32x32x3 -- sneaker, ankle_boot, bag, sandal, trouser and
                          the rest. NOT in CIFAR-10 or CIFAR-100 at all: a different corpus, a different domain, and
                          a distinctive asymmetric silhouette rather than the keyboard's repetitive texture
    flowers102            Flowers-102 photographs downsampled to 32x32 (a different corpus, not just a new label)

Controls in every run: certificate sanity (||CH||, quotient form, rank C, excitation gap); residual at the truths
against public images and random chart points; the blend fraction of every found image in span{phi(x_i)}; landing by
IMAGE ERROR (not SSIM); the attacker's own ranking (top-20 starts by residual); a same-class control using the same
max-over-starts statistic; and --wrong_release (certificate from a release trained on 8 OTHER images of the same
new class -- must give zero landings).

  python -u -m experiments.cifar.cifar_newclass --arch mlp --newclass cifar100:keyboard --k 32
  python -u -m experiments.cifar.cifar_newclass --arch cnn --newclass flowers102 --k 32 --wrong_release
"""
import argparse, glob, json, math, os, pickle, socket, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


def log(s): print(s, flush=True)


# --------------------------------------------------------------------------------------------- data
def load_cifar10(root):
    d = os.path.join(root, "cifar-10-batches-py"); Xs, ys = [], []
    for i in range(1, 6):
        b = pickle.load(open(os.path.join(d, f"data_batch_{i}"), "rb"), encoding="bytes"); Xs.append(b[b"data"]); ys += b[b"labels"]
    b = pickle.load(open(os.path.join(d, "test_batch"), "rb"), encoding="bytes")
    return (np.concatenate(Xs).astype(np.float32) / 255.0, np.array(ys, dtype=np.int64),
            b[b"data"].astype(np.float32) / 255.0, np.array(b[b"labels"], dtype=np.int64))


def load_cifar100_class(root, name):
    meta = pickle.load(open(os.path.join(root, "cifar-100-python", "meta"), "rb"), encoding="bytes")
    names = [n.decode() for n in meta[b"fine_label_names"]]
    idx = int(name) if str(name).isdigit() else names.index(name)
    out = {}
    for split in ("train", "test"):
        b = pickle.load(open(os.path.join(root, "cifar-100-python", split), "rb"), encoding="bytes")
        lab = np.array(b[b"fine_labels"]); X = b[b"data"].astype(np.float32) / 255.0
        out[split] = X[lab == idx]
    return out, names[idx]


FASHION = ["t_shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot"]


def load_fashion(root, name):
    """A FashionMNIST class as a CIFAR-shaped added-on class: 28x28 grey -> 32x32, replicated to 3 channels,
       flattened channel-major so it drops straight into the CIFAR-10 pipeline. Nothing in CIFAR-10 or CIFAR-100 is
       footwear, a bag or a garment, so this is a class from OUTSIDE the corpus the backbone was built from -- and
       unlike a keyboard it has a distinctive asymmetric silhouette that a reader can identify at a glance."""
    import torch.nn.functional as Fn
    idx = FASHION.index(name) if not str(name).isdigit() else int(name)
    d = os.path.join(root, "FashionMNIST", "raw")
    out = {}
    for split, ip, lp in (("train", "train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
                          ("test", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")):
        with open(os.path.join(d, ip), "rb") as f:
            f.read(16); img = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
        with open(os.path.join(d, lp), "rb") as f:
            f.read(8); lab = np.frombuffer(f.read(), dtype=np.uint8)
        sel = img[lab == idx].astype(np.float32) / 255.0
        t = torch.tensor(sel)[:, None]                                                  # (n,1,28,28)
        t = Fn.interpolate(t, size=(32, 32), mode="bilinear", align_corners=False)
        out[split] = t.expand(-1, 3, -1, -1).reshape(len(t), -1).numpy().astype(np.float64)
    return out, f"fashion_{FASHION[idx]}"


def load_flowers102(root, n_public=2000, n_private=64, seed=1):
    """Flowers-102 jpgs downsampled to 32x32 RGB, flattened like CIFAR (channel-major)."""
    from PIL import Image
    files = sorted(glob.glob(os.path.join(root, "flowers-102", "jpg", "*.jpg")))
    rng = np.random.RandomState(seed); rng.shuffle(files)
    take = files[: n_public + n_private]
    X = np.stack([np.asarray(Image.open(f).convert("RGB").resize((32, 32), Image.BILINEAR), dtype=np.float32).transpose(2, 0, 1).reshape(-1) / 255.0 for f in take])
    return {"train": X[n_private:], "test": X[:n_private]}, "flowers102"


# --------------------------------------------------------------------------------------------- model
class MLP(nn.Module):
    """The project's structure (Haim et al. / `train_cifar_backbone.py`): 3072-1000-1000-10, GELU, bias on layer 1 only.
       phi = the penultimate activations (n = 1000), head = the output layer (m = 10 -> extended to 11)."""
    def __init__(s, m=10):
        super().__init__()
        s.l1 = nn.Linear(3072, 1000); s.l2 = nn.Linear(1000, 1000, bias=False); s.head = nn.Linear(1000, m, bias=False)
    def phi(s, x): return F.gelu(s.l2(F.gelu(s.l1(x.reshape(len(x), -1)))))
    def forward(s, x, W=None): return F.linear(s.phi(x), s.head.weight if W is None else W)


class CNN(nn.Module):
    """Conv stack -> 256-dim penultimate -> 10-way head. The head's INPUT is what the certificate acts on."""
    def __init__(s, m=10):
        super().__init__()
        def blk(i, o): return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU()]
        s.body = nn.Sequential(*blk(3, 64), *blk(64, 64), nn.MaxPool2d(2), *blk(64, 128), *blk(128, 128), nn.MaxPool2d(2),
                               *blk(128, 256), *blk(256, 256), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(256 * 16, 256), nn.GELU())
        s.head = nn.Linear(256, m, bias=False)
    def phi(s, x): return s.body(x.reshape(-1, 3, 32, 32))
    def forward(s, x, W=None): return F.linear(s.phi(x), s.head.weight if W is None else W)


def train_backbone(path, root, dev, epochs, gate, train_gate, arch, overtrain=False, loss_tol=1e-4):
    """Train once and cache. Reports train and test accuracy so 'fully trained' is a number, not an assumption."""
    Net = MLP if arch == "mlp" else CNN
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        net = Net().to(dev); net.load_state_dict(blob["state_dict"]); net.eval()
        log(f"# backbone loaded from {path}: train {blob['train_acc']*100:.2f}%, test {blob['test_acc']*100:.1f}%, "
            f"train loss {blob.get('train_loss', float('nan')):.2e}, median margin {blob.get('margin_median', float('nan')):.2f} ({blob['epochs']} epochs{', OVER-TRAINED' if blob.get('overtrain') else ''})")
        return net, blob["test_acc"], blob["train_acc"]
    Xtr, ytr, Xte, yte = load_cifar10(root)
    Xtr_t = torch.tensor(Xtr).to(dev); ytr_t = torch.tensor(ytr).to(dev); Xte_t = torch.tensor(Xte).to(dev); yte_t = torch.tensor(yte).to(dev)
    net = Net().to(dev)
    if overtrain:                                            # Haim et al.'s regime: no augmentation, NO weight decay, train past zero error
        opt = torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0, nesterov=True); sched = None
    else:
        opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.05, epochs=epochs, steps_per_epoch=math.ceil(len(Xtr_t) / 256))
    for ep in range(epochs):
        net.train()
        for i in torch.randperm(len(Xtr_t), device=dev).split(256):
            xb = Xtr_t[i].reshape(-1, 3, 32, 32); yb = ytr_t[i]
            if arch == "cnn" and not overtrain:                                            # augmentation for the conv arm only: the MLP
                if torch.rand(1).item() < 0.5: xb = xb.flip(3)           # arm must CONVERGE on its train set (train_gate)
                p = F.pad(xb, (4, 4, 4, 4), mode="reflect"); dx, dy = torch.randint(0, 9, (2,))
                xb = p[:, :, dy:dy + 32, dx:dx + 32]
            opt.zero_grad(); F.cross_entropy(net(xb.reshape(len(xb), -1)), yb).backward(); opt.step()
            if sched is not None: sched.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                zt = torch.cat([net(Xtr_t[i:i + 1000]) for i in range(0, len(Xtr_t), 1000)])
                tr_acc = float((zt.argmax(1) == ytr_t).float().mean()); tr_loss = float(F.cross_entropy(zt, ytr_t))
                te = float((torch.cat([net(Xte_t[i:i + 1000]).argmax(1) for i in range(0, len(Xte_t), 1000)]) == yte_t).float().mean())
            log(f"#   epoch {ep+1}/{epochs}: train {tr_acc*100:.2f}% (loss {tr_loss:.2e}), test {te*100:.2f}%")
            if overtrain and tr_acc == 1.0 and tr_loss < loss_tol:
                log(f"#   over-trained: 100% train accuracy and loss {tr_loss:.2e} < {loss_tol:.0e} -- stopping at epoch {ep+1}"); break
    net.eval()
    with torch.no_grad():
        te = float((torch.cat([net(Xte_t[i:i + 1000]).argmax(1) for i in range(0, len(Xte_t), 1000)]) == yte_t).float().mean())
        zt = torch.cat([net(Xtr_t[i:i + 1000]) for i in range(0, len(Xtr_t), 1000)])
        tr = float((zt.argmax(1) == ytr_t).float().mean()); tr_loss = float(F.cross_entropy(zt, ytr_t))
        zy = zt[torch.arange(len(zt), device=dev), ytr_t]; zo = zt.clone(); zo[torch.arange(len(zt), device=dev), ytr_t] = -float("inf")
        mar = zy - zo.max(1).values; mar_med = float(mar.median()); mar_min = float(mar.min())
    log(f"# train loss {tr_loss:.3e}; margins on the training set: median {mar_med:.2f}, min {mar_min:.2f}, fraction negative {(mar < 0).float().mean()*100:.2f}%")
    log(f"# backbone ({arch}) trained: train {tr*100:.1f}%, test {te*100:.1f}%   test gate {'PASS' if te >= gate else 'FAIL'} (>= {gate*100:.0f}%), "
        f"train gate {'PASS' if tr >= train_gate else 'FAIL'} (>= {train_gate*100:.0f}%)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(state_dict=net.state_dict(), test_acc=te, train_acc=tr, train_loss=tr_loss, margin_median=mar_med, margin_min=mar_min, overtrain=overtrain, epochs=epochs), path)
    return net, te, tr


def lm_cert(fun, w0, iters=300, lam=1e-2):
    """certificate.lm_cert, verbatim in behaviour: LM on w -> C phi(psi(w)) / ||A_T phi||, autograd Jacobian."""
    import torch.func as tfn
    w = w0.clone(); f = fun(w); obj = float(f @ f)
    for it in range(iters):
        J = tfn.jacfwd(fun)(w)
        for _ in range(12):
            step = torch.linalg.solve(J.T @ J + lam * torch.eye(J.shape[1], device=w.device, dtype=w.dtype), -(J.T @ f))
            wn = w + step; fn_ = fun(wn); on = float(fn_ @ fn_)
            if on < obj: w, f, obj = wn, fn_, on; lam = max(lam / 3, 1e-15); break
            lam *= 4
        else: return w, obj, it + 1
        if obj < 1e-30: return w, obj, it + 1
    return w, obj, iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--newclass", default="cifar100:keyboard", help="cifar100:<name|idx> or flowers102")
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=64); ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--starts", type=int, default=200); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--private", choices=["onchart", "raw"], default="onchart")
    ap.add_argument("--wrong_release", action="store_true")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--arch", choices=["mlp", "cnn"], default="mlp")
    ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--gate", type=float, default=None,
                    help="minimum TEST accuracy before any attack number is produced (default: 0.50 for the mlp -- a pixel MLP's ceiling -- and 0.80 for the cnn)")
    ap.add_argument("--overtrain", action="store_true", help="the original paper's regime: no augmentation, no weight decay, train past zero error until the loss collapses")
    ap.add_argument("--loss-tol", type=float, default=1e-4, help="--overtrain stops when train accuracy is 100%% and the train loss is below this")
    ap.add_argument("--train-gate", type=float, default=None, help="minimum TRAIN accuracy: 'fully trained' means converged on its own training set. The gate exists to catch an UNTRAINED backbone, not to demand memorisation; the pixel MLP reaches 93.6% train / 57.9% test and the conv net ~99% / ~88%.")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data-root", default="data"); ap.add_argument("--flowers-root", default="data")
    ap.add_argument("--fashion-root", default="dataset_reconstruction/data")
    ap.add_argument("--degrade28", action="store_true", help="RESOLUTION CONTROL: put the public and private images through "
                    "32 -> 28 -> 32 bilinear, the same path the FashionMNIST classes travel. The off-corpus cells differ from "
                    "the CIFAR-100 cells in resolution history as well as in domain, and this cell removes that confound so a "
                    "difference can be attributed to domain alone.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default="results/cifar_newclass"); ap.add_argument("--fig-dir", default="figures/cifar_newclass")
    a = ap.parse_args(); dev = torch.device(a.device)
    os.makedirs(a.save_dir, exist_ok=True); os.makedirs(a.fig_dir, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    if a.gate is None: a.gate = 0.50 if a.arch == "mlp" else 0.80
    if a.epochs is None: a.epochs = (200 if a.overtrain else 60) if a.arch == "mlp" else (120 if a.overtrain else 40)
    if a.train_gate is None: a.train_gate = 0.999 if a.overtrain else 0.90
    if a.ckpt is None: a.ckpt = f"models/exact_inversion/cifar10_{a.arch}{'_overtrained' if a.overtrain else ''}_newclass.pth"
    net, te, tr = train_backbone(a.ckpt, a.data_root, dev, a.epochs, a.gate, a.train_gate, a.arch, a.overtrain, a.loss_tol)
    if te < a.gate or tr < a.train_gate:
        log("# GATE FAILED -- stopping before any attack number is produced."); return
    net = net.double()
    for p_ in net.parameters(): p_.requires_grad_(False)

    if a.newclass.startswith("cifar100:"): pool, cname = load_cifar100_class(a.data_root, a.newclass.split(":", 1)[1])
    elif a.newclass.startswith("fashion:"): pool, cname = load_fashion(a.fashion_root, a.newclass.split(":", 1)[1])
    elif a.newclass == "flowers102": pool, cname = load_flowers102(a.flowers_root, seed=a.seed)
    else: raise ValueError(a.newclass)
    Pub = torch.tensor(pool["train"], dtype=torch.float64, device=dev); Pri = torch.tensor(pool["test"], dtype=torch.float64, device=dev)
    if a.degrade28:
        rt = lambda X: F.interpolate(F.interpolate(X.reshape(-1, 3, 32, 32), size=(28, 28), mode="bilinear", align_corners=False),
                                     size=(32, 32), mode="bilinear", align_corners=False).reshape(len(X), -1)
        Pub, Pri = rt(Pub), rt(Pri); cname = cname + "_28roundtrip"
        log(f"# RESOLUTION CONTROL: public and private images sent through 32 -> 28 -> 32, matching the footwear cells' path")
    g = torch.Generator().manual_seed(a.seed + 7); sel = torch.randperm(Pri.shape[0], generator=g)[: 2 * a.N]
    X_raw = Pri[sel[: a.N]].T.contiguous(); X_other = Pri[sel[a.N:]].T.contiguous()          # the wrong-release control's images
    m, n = 11, net.head.weight.shape[1]
    W0 = torch.cat([net.head.weight.double(), torch.zeros(1, n, dtype=torch.float64, device=dev)], 0)   # 11th row zeroed (the letters-cell practice)
    phi = lambda X: net.phi(X.T).T                                                            # (3072, B) -> (n, B); net and X are both float64
    log(f"# cifar_newclass  arch={a.arch} backbone test {te*100:.1f}% train {tr*100:.1f}%  new class '{cname}' (public {Pub.shape[0]}, private {a.N} held out)  m={m} n={n} r={a.r} k={a.k} private={a.private}  host={socket.gethostname()}")
    with torch.no_grad(): log(f"# the trained model reads the private images as CIFAR-10 classes: {(W0 @ phi(X_raw)).argmax(0).tolist()}")

    mean = Pub.mean(0); U_, S_, Vh_ = torch.linalg.svd(Pub - mean, full_matrices=False); V = Vh_[: a.k].T.contiguous()
    psi = lambda W: mean[:, None] + V @ W; coords = lambda X: V.T @ (X - mean[:, None])
    X_on = psi(coords(X_raw)); X_train = X_on if a.private == "onchart" else X_raw
    X_rel = (psi(coords(X_other)) if a.private == "onchart" else X_other) if a.wrong_release else X_train
    coord_std = coords(Pub[:5000].T).std(dim=1, keepdim=True)
    repr_err = torch.linalg.norm(X_on - X_raw, dim=0) / torch.linalg.norm(X_raw, dim=0)
    log(f"# PCA chart k={a.k}: explained variance {float((S_[:a.k]**2).sum()/(S_**2).sum()):.3f}; representation error of the privates {[f'{v:.3f}' for v in repr_err.tolist()]}")

    H = phi(X_rel); y = torch.full((a.N,), 10, device=dev)
    A0 = (1.0 / math.sqrt(n) * torch.randn(a.r, n, generator=torch.Generator().manual_seed(a.seed + 7), dtype=torch.float64)).to(dev)
    A, B = A0.clone(), torch.zeros(m, a.r, dtype=torch.float64, device=dev)
    Y = torch.eye(m, device=dev, dtype=torch.float64)[y].T
    for t in range(a.T):
        z = W0 @ H + B @ (A @ H); D = (torch.softmax(z, 0) - Y) / a.N
        B, A = B - a.lr * (D @ (A @ H).T), A - a.lr * (B.T @ D @ H.T)
    A_T, B_T = A, B
    sB = torch.linalg.svdvals(B_T); Np = int((sB > a.tol * sB[0]).sum())
    _, _, VhB = torch.linalg.svd(B_T, full_matrices=False); Q = VhB[:Np].T
    C = A_T - Q @ (Q.T @ A_T)
    Hp = phi(X_train)                                                                          # the ORIGINAL privates' features (scoring)
    san = dict(CH_rel=float((C @ H).norm() / (C.norm() * H.norm())), rank_C=int(torch.linalg.matrix_rank(C, rtol=1e-10)),
               quotient=float((C - (A0 - Q @ (Q.T @ A0))).norm() / C.norm()), gap=float(sB[Np - 1] / sB[Np]), n_prime=Np)
    log(f"# release: rank B_T = {Np} (expect {a.N}), rank C = {san['rank_C']} (expect {a.r - Np}), gap {san['gap']:.1e}, ||CH||/(||C||||H||) = {san['CH_rel']:.1e}, quotient form {san['quotient']:.1e}")
    obj_of = lambda X: (torch.linalg.norm(C @ phi(X), dim=0) / torch.linalg.norm(A_T @ phi(X), dim=0))
    with torch.no_grad():
        res_truth = obj_of(X_train); res_on = obj_of(X_on); res_pub = obj_of(Pub[:200].T); res_rand = obj_of(psi(torch.randn(a.k, 200, generator=torch.Generator().manual_seed(3), dtype=torch.float64).to(dev) * coord_std))
        Qh, _ = torch.linalg.qr(Hp)
    log(f"# residual at the private training inputs {[f'{v:.1e}' for v in res_truth.tolist()]}; at the chart projections {[f'{v:.1e}' for v in res_on.tolist()]}")
    log(f"# residual on 200 public images of the same class: median {res_pub.median():.2e}, min {res_pub.min():.2e}; on 200 random chart points: median {res_rand.median():.2e}, min {res_rand.min():.2e}")

    def fun(w):
        f = phi(psi(w.reshape(a.k, 1)))
        return (C @ f).reshape(-1) / torch.linalg.norm(A_T @ f)
    with torch.no_grad(): feat_ref = float(torch.linalg.norm(A_T @ phi(Pub[:256].T), dim=0).median())
    gs = torch.Generator().manual_seed(a.seed + 31); t0 = time.time(); runs = []; Ws = []
    for s in range(a.starts):
        w0 = (torch.randn(a.k, 1, generator=gs).to(dev).double() * coord_std).reshape(-1)
        w, obj, it = lm_cert(fun, w0, a.iters)
        with torch.no_grad():
            x = psi(w.reshape(a.k, 1))[:, 0]
            e = [float(torch.linalg.norm(x - X_train[:, i]) / torch.linalg.norm(X_train[:, i])) for i in range(a.N)]
            fr = float(torch.linalg.norm(A_T @ phi(x.reshape(-1, 1))) / feat_ref)
            bl = float((Qh.T @ phi(x.reshape(-1, 1))).norm() ** 2 / phi(x.reshape(-1, 1)).norm() ** 2)
        j = int(np.argmin(e)); runs.append(dict(objective=obj, iters=it, nearest=j, err=e[j], landed=bool(e[j] < 1e-2), degenerate=bool(fr < 0.05), blend=bl)); Ws.append(w.detach().cpu())
        if (s + 1) % 25 == 0: log(f"   {s+1}/{a.starts} starts, {time.time()-t0:.0f}s, landed {sum(r_['landed'] for r_ in runs)}, best objective {min(r_['objective'] for r_ in runs):.1e}")
    W = torch.stack(Ws, 1).to(dev); X_found = psi(W)
    import kornia.metrics as km
    ssim = lambda A_, B_: float(km.ssim(A_.reshape(1, 3, 32, 32).clamp(0, 1).float(), B_.reshape(1, 3, 32, 32).clamp(0, 1).float(), window_size=3).mean())
    valid = [r_ for r_ in runs if not r_["degenerate"]]
    order = sorted(range(len(runs)), key=lambda s: (runs[s]["objective"] if not runs[s]["degenerate"] else float("inf")))
    pub_proj = psi(coords(Pub[:200].T)); nearest_pub = pub_proj[:, torch.cdist(X_train.T, pub_proj.T).argmin(1)]
    per = []; best_imgs = []
    for i in range(a.N):
        mine = [s for s in range(len(runs)) if runs[s]["nearest"] == i]
        land = sum(1 for s in mine if runs[s]["landed"])
        if mine:
            sb = min(mine, key=lambda s: runs[s]["err"]); xb = X_found[:, sb]
            per.append(dict(i=i, landings=land, best_err=runs[sb]["err"], best_objective=runs[sb]["objective"], blend=runs[sb]["blend"],
                            ssim_vs_raw=ssim(xb, X_raw[:, i]), ssim_vs_train=ssim(xb, X_train[:, i]),
                            control_ssim_max=max(ssim(X_found[:, s], nearest_pub[:, i]) for s in range(0, len(runs), max(1, len(runs) // 60)))))
            best_imgs.append(xb.cpu())
        else:
            per.append(dict(i=i, landings=land, best_err=None, best_objective=None, blend=None, ssim_vs_raw=None, ssim_vs_train=None, control_ssim_max=None)); best_imgs.append(torch.zeros(3072))
        per[-1].update(chart_floor_err=float(repr_err[i]), chart_floor_ssim=ssim(X_on[:, i], X_raw[:, i]), res_truth=float(res_truth[i]), res_projection=float(res_on[i]))
    row = dict(part="cifar_newclass", arch=a.arch, overtrained=a.overtrain, newclass=a.newclass, class_name=cname, backbone_test_acc=te, backbone_train_acc=tr, private=a.private, wrong_release=a.wrong_release,
               k=a.k, r=a.r, N=a.N, m=m, n=n, T=a.T, lr=a.lr, seed=a.seed, tol=a.tol, sanity=san, cert_line=a.r - Np, capacity_line=m + a.r - a.N,
               residual_at_truth=res_truth.tolist(), residual_at_projection=res_on.tolist(), res_public_median=float(res_pub.median()), res_public_min=float(res_pub.min()),
               res_chart_random_median=float(res_rand.median()), starts=len(runs), landed=sum(r_["landed"] for r_ in runs), landings_per_image=[p_["landings"] for p_ in per],
               images_found=sum(1 for p_ in per if p_["landings"] > 0), top20_by_residual_landed=[int(runs[s]["landed"]) for s in order[:20]],
               blend_found_median=float(np.median([r_["blend"] for r_ in runs])), objective_median=float(np.median([r_["objective"] for r_ in runs])),
               objective_min=float(min(r_["objective"] for r_ in valid)) if valid else None, n_degenerate=len(runs) - len(valid),
               chart_repr_err=repr_err.tolist(), per_image=per, seconds=time.time() - t0, host=socket.gethostname(), cmd=" ".join(sys.argv))
    log("\n=== RESULT ===")
    log(f"landed {row['landed']}/{row['starts']}, per image {row['landings_per_image']}, found {row['images_found']}/{a.N}; top-20 by residual landed {sum(row['top20_by_residual_landed'])}/20")
    log(f"closest approach per image {[None if p_['best_err'] is None else round(p_['best_err'], 4) for p_ in per]}")
    log(f"SSIM vs raw: attack {[None if p_['ssim_vs_raw'] is None else round(p_['ssim_vs_raw'], 2) for p_ in per]} | chart floor {[round(p_['chart_floor_ssim'], 2) for p_ in per]} | control {[None if p_['control_ssim_max'] is None else round(p_['control_ssim_max'], 2) for p_ in per]}")
    log(f"blend fraction of found images (median) {row['blend_found_median']:.3f}")
    print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    tag = f"{a.arch}{'_overtrained' if a.overtrain else ''}_{cname}_k{a.k}_{a.private}{'_wrongrelease' if a.wrong_release else ''}"
    torch.save(dict(x_raw=X_raw.cpu(), x_chart=X_on.cpu(), x_train=X_train.cpu(), x_found_best=torch.stack(best_imgs, 1), A_T=A_T.cpu(), B_T=B_T.cpu(), A0=A0.cpu(), C=C.cpu(), W=W.cpu(), runs=runs, meta=row),
               os.path.join(a.save_dir, f"{tag}.pth"))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    panels = [("private (raw)", X_raw.cpu()), ("chart projection (public PCA)", X_on.cpu()), ("closest random start (certificate only)", torch.stack(best_imgs, 1))]
    fig, ax = plt.subplots(3, a.N, figsize=(1.35 * a.N + 1.8, 4.9)); fig.subplots_adjust(left=0.16, top=0.83, bottom=0.02, hspace=0.3)
    for ri, (name, imgs) in enumerate(panels):
        for j in range(a.N):
            ax[ri, j].imshow(imgs[:, j].reshape(3, 32, 32).permute(1, 2, 0).clamp(0, 1).float().numpy()); ax[ri, j].axis("off")
            if ri == 2: ax[ri, j].set_title(("landed" if per[j]["landings"] else f"err {per[j]['best_err']:.2f}" if per[j]["best_err"] else "-") + f"\nssim {per[j]['ssim_vs_raw']:.2f}" if per[j]["ssim_vs_raw"] else "-", fontsize=6.5)
        p_ = ax[ri, 0].get_position(); fig.text(0.012, (p_.y0 + p_.y1) / 2, name, fontsize=8, va="center")
    fig.suptitle(f"'{cname}' as an 11th class on a{'n OVER-TRAINED' if a.overtrain else ''} CIFAR-10 {a.arch.upper()} ({te*100:.1f}% test / {tr*100:.2f}% train) -- head LoRA r={a.r}, T={a.T} SGD steps (lr={a.lr}), public PCA chart k={a.k}, {a.private} privates"
                 f"{', WRONG-RELEASE CONTROL' if a.wrong_release else ''}\n{row['landed']}/{row['starts']} random starts landed, {row['images_found']}/{a.N} images found, "
                 f"residual at truths {max(row['residual_at_truth']):.0e} vs public median {row['res_public_median']:.1e}", fontsize=9)
    fig.savefig(os.path.join(a.fig_dir, f"{tag}.png"), dpi=150); plt.close(fig)
    log(f"saved {a.save_dir}/{tag}.pth and {a.fig_dir}/{tag}.png")


if __name__ == "__main__":
    main()
