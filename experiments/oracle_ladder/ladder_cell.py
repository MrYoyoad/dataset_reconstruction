#!/usr/bin/env python3
"""ONE cell of the oracle-chart ladder, on the two examples the meeting document uses, plus the MNIST letters release.

New code; no canonical script is modified. Each invocation runs a single (example, chart) cell and writes one JSON
row plus its tensors, so every cell gets the SAME solver and the SAME 400 starts and nothing is tuned between them --
and so the convolutional example, at roughly 13 s per start, can run its charts in parallel rather than in series.

  examples   mlp_motorcycle    : eight CIFAR-100 motorcycles added to the CIFAR-10 MLP
             cnn_keyboard      : eight CIFAR-100 keyboards added to the CIFAR-10 CNN
             mlp_letter_a      : eight EMNIST letters "a" (test split) added to the 98% MNIST MLP (mnist_mlp_strong.pth)
             mlp_letter_a_full : the same on the over-trained twin mnist_mlp_strong_full.pth, if it exists (WP0 gate)
  privates   RAW images, not their chart projections -- so the release is fine-tuned on the images and the attack
             targets the images.  The MNIST privates are the SAME eight letters as the existing letters cells
             (new_class.py / ntk_vs_certificate.py: randperm of the letter's test split under seed+7, first N); the
             join-key indices into that split are printed and stored in the row.
  charts     oracle:<eps>  the ladder construction, VERBATIM from experiments/cifar/cifar_charts.py: perturb each
                           private image by relative noise eps, span those perturbed vectors, fill the rest of
                           the k directions with public PCA, orthonormalise.  NOT ATTACKER-AVAILABLE at any eps,
                           including eps = 0.
             pca           the actual public PCA chart of the added class.  The ONLY attacker-available chart here.

eps is NOT the same quantity as the chart's projection error on the true images: eps perturbs the images before
the span is taken, so the true images do not lie in the resulting chart and their projection error is measured
separately. Both are reported and the figures are drawn against the MEASURED error.

Every row records the base checkpoint path and its train / test accuracy and train loss (WP0 of the 2026-09-18 plan);
where the checkpoint dict lacks them they are measured at load time on the full train split.

  python -m experiments.oracle_ladder.ladder_cell --example mlp_motorcycle --chart oracle:0.05
  python -m experiments.oracle_ladder.ladder_cell --example cnn_keyboard --chart pca
  python -m experiments.oracle_ladder.ladder_cell --example mlp_letter_a --chart oracle:0 --wrong_release
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch

from experiments.cifar.cifar_newclass import train_backbone, load_cifar100_class, lm_cert

torch.set_default_dtype(torch.float64)

EXAMPLES = {
    "mlp_motorcycle":    dict(dataset="cifar", arch="mlp", cls="motorcycle", ckpt="models/exact_inversion/cifar10_mlp_newclass.pth",
                              gate=(0.50, 0.90), epochs=60, shape=(3, 32, 32), data_root="data"),
    "cnn_keyboard":      dict(dataset="cifar", arch="cnn", cls="keyboard",   ckpt="models/exact_inversion/cifar10_cnn_newclass.pth",
                              gate=(0.80, 0.90), epochs=40, shape=(3, 32, 32), data_root="data"),
    "mlp_letter_a":      dict(dataset="mnist", arch="mlp", cls="a", ckpt="models/exact_inversion/mnist_mlp_strong.pth",
                              shape=(1, 28, 28), data_root="dataset_reconstruction/data"),
    "mlp_letter_a_full": dict(dataset="mnist", arch="mlp", cls="a", ckpt="models/exact_inversion/mnist_mlp_strong_full.pth",
                              shape=(1, 28, 28), data_root="dataset_reconstruction/data"),
    # arm (b) of WP5 (plan audit 2026-09-18): the depth window's own encoder, images and chart pool. Head adapter ONLY on the
    # 15-layer MNIST MLP; privates = the eight TEST digits at the k-sweep's join-key indices (real_encoder_ranklaw.py:
    # Generator seed+7, randperm of the test set, first N) with their TRUE labels -- a confident batch of known classes, so
    # the head is NOT extended and the recording strength (||B_T||_F, spectrum, per-image softmax residual) is in the row.
    "d15_digits":        dict(dataset="mnist_digits", arch="mlp_d15", cls="digits", ckpt="models/exact_inversion/mnist_mlp_d15w1000.pth",
                              shape=(1, 28, 28), data_root="dataset_reconstruction/data", n_fit=50000,
                              base_gate_note="d15 FAILED the WP0 base gate (98.69% train, CE 4.9e-2 at the time of the plan audit); used "
                                             "UNCHANGED because the depth window (real_encoder_ranklaw) was measured on it"),
}
WP0_TRAIN_ACC, WP0_TRAIN_LOSS = 0.995, 1e-2          # "fully trained" per notes/plan_2026-09-18 WP0


def log(s): print(s, flush=True)


def ssim(a, b, shape):
    import kornia.metrics as km
    return float(km.ssim(a.reshape(1, *shape).clamp(0, 1).float(), b.reshape(1, *shape).clamp(0, 1).float(), window_size=3).mean())


def acc_loss_of(logits_fn, X, y, dev):
    """Accuracy and cross-entropy of a (784,B)->(m,B) logits map over a full split, chunked, FP64."""
    Xt = torch.tensor(X, device=dev); yt = torch.tensor(y, device=dev); Z = []
    with torch.no_grad():
        for i in range(0, len(Xt), 5000): Z.append(logits_fn(Xt[i:i + 5000].T).T)
        Z = torch.cat(Z); return float((Z.argmax(1) == yt).double().mean()), float(torch.nn.functional.cross_entropy(Z, yt))


def load_example(ex, root, dev):
    """Returns phi (D,B)->(n,B), the frozen head W_head (m0 x n), Pub / Pri image matrices (rows), class name, the
       backbone record dict(ckpt, test_acc, train_acc, train_loss), and the labels of Pri (None = added class, the head is
       extended by a zero row and every private carries the new label). Nothing here is tuned per example."""
    if ex["dataset"] == "mnist_digits":
        from experiments.exact_inversion.trained_backbone import read_idx
        from experiments.exact_inversion.deep_stack import inputs_of, load_deep
        Ws, b1, ck = load_deep(ex["ckpt"], dev)
        phi = lambda X: inputs_of(X, Ws, b1)[-1]                               # the 1000-d INPUT to the head, as in the k-sweep
        W_head = Ws[-1]
        Xtr, ytr = read_idx(root, "train"); Xte, yte = read_idx(root, "test")
        tr, tr_loss = acc_loss_of(lambda X: W_head @ phi(X), Xtr, ytr, dev); te, _ = acc_loss_of(lambda X: W_head @ phi(X), Xte, yte, dev)
        log(f"# deep stack {ex['ckpt']}: depth {len(Ws)}, width {Ws[0].shape[0]}, checkpoint test_acc {ck.get('test_acc')}")
        Pub = torch.tensor(Xtr[: ex["n_fit"]], device=dev); Pri = torch.tensor(Xte, device=dev)   # the k-sweep's chart pool and test set
        return phi, W_head, Pub, Pri, ex["cls"], dict(ckpt=ex["ckpt"], test_acc=te, train_acc=tr, train_loss=tr_loss), torch.tensor(yte, device=dev)
    if ex["dataset"] == "cifar":
        net, te, tr = train_backbone(ex["ckpt"], root, dev, ex["epochs"], ex["gate"][0], ex["gate"][1], ex["arch"])
        net = net.double()
        for p_ in net.parameters(): p_.requires_grad_(False)
        blob = torch.load(ex["ckpt"], map_location="cpu", weights_only=False)
        pool, cname = load_cifar100_class(root, ex["cls"])
        Pub = torch.tensor(pool["train"], dtype=torch.float64, device=dev); Pri = torch.tensor(pool["test"], dtype=torch.float64, device=dev)
        return (lambda X: net.phi(X.T).T), net.head.weight.double(), Pub, Pri, cname, \
            dict(ckpt=ex["ckpt"], test_acc=float(te), train_acc=float(tr), train_loss=float(blob.get("train_loss", float("nan")))), None
    # ---- MNIST: the 98% MLP (784-1000-1000-10, GELU) as loaded by every existing letters cell
    from experiments.exact_inversion.trained_backbone import TrainedBackbone, read_idx
    from experiments.exact_inversion.new_class import load_emnist_letters
    bb = TrainedBackbone(ex["ckpt"], dev, "gelu")
    blob = torch.load(ex["ckpt"], map_location="cpu", weights_only=False)
    Xtr, ytr = read_idx(root, "train"); Xte, yte = read_idx(root, "test")
    tr, tr_loss = acc_loss_of(bb.logits, Xtr, ytr, dev); te, _ = acc_loss_of(bb.logits, Xte, yte, dev)
    if "train_acc" in blob: log(f"# checkpoint records train_acc {blob['train_acc']*100:.2f}%; measured now {tr*100:.2f}% (the row carries the measurement)")
    if "test_acc" in blob: log(f"# checkpoint records test_acc {blob['test_acc']*100:.2f}%; measured now {te*100:.2f}%")
    fl = load_emnist_letters(root, ex["cls"])
    Pub = torch.tensor(fl["train"][0], device=dev); Pri = torch.tensor(fl["test"][0], device=dev)
    return bb.phi, bb.W0, Pub, Pri, f"letter_{ex['cls']}", dict(ckpt=ex["ckpt"], test_acc=te, train_acc=tr, train_loss=tr_loss), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", choices=list(EXAMPLES), required=True)
    ap.add_argument("--chart", required=True, help="pca | oracle:<eps>")
    ap.add_argument("--wrong_release", action="store_true", help="control: the release is trained on eight OTHER images of the same class")
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=64); ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--starts", type=int, default=400); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--data-root", default=None, help="default per example: data (CIFAR) / dataset_reconstruction/data (MNIST)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="results/oracle_ladder")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(a.out_dir, exist_ok=True)
    ex = EXAMPLES[a.example]; shape = ex["shape"]; root = a.data_root or ex["data_root"]
    is_oracle = a.chart.startswith("oracle")
    eps = float(a.chart.split(":", 1)[1]) if is_oracle else float("nan")

    phi, W_head, Pub, Pri, cname, bbrec, ylab = load_example(ex, root, dev)
    te, tr = bbrec["test_acc"], bbrec["train_acc"]
    wp0 = bool(tr >= WP0_TRAIN_ACC and bbrec["train_loss"] <= WP0_TRAIN_LOSS)
    log(f"# backbone {bbrec['ckpt']}: train {tr*100:.2f}% (loss {bbrec['train_loss']:.2e}), test {te*100:.2f}%  "
        f"WP0 fully-trained gate (train >= {WP0_TRAIN_ACC*100:.1f}% and loss <= {WP0_TRAIN_LOSS:.0e}): {'PASS' if wp0 else 'FAIL'}")
    if ex.get("base_gate_note"): log(f"# NOTE: {ex['base_gate_note']}")
    g = torch.Generator().manual_seed(a.seed + 7); perm = torch.randperm(Pri.shape[0], generator=g)
    X_raw = Pri[perm[: a.N]].T.contiguous()                                        # the SAME eight images as the existing cells
    X_other = Pri[perm[a.N: 2 * a.N]].T.contiguous()                               # for the wrong-release control
    join_idx = [int(v) for v in perm[: a.N]]; other_idx = [int(v) for v in perm[a.N: 2 * a.N]]
    log(f"# private join-key indices into the {cname} test split (seed {a.seed}+7, first {a.N}): {join_idx}"
        f"   wrong-release images: {other_idx}")
    n = W_head.shape[1]
    if ylab is None:                                                               # added class: extend the head by a zero row
        m = W_head.shape[0] + 1
        W0 = torch.cat([W_head, torch.zeros(1, n, dtype=torch.float64, device=dev)], 0)
        y_raw = torch.full((a.N,), m - 1, device=dev); y_other = y_raw.clone()
    else:                                                                          # known classes: the head as trained, true labels
        m = W_head.shape[0]; W0 = W_head
        y_raw = ylab[perm[: a.N].to(dev)]; y_other = ylab[perm[a.N: 2 * a.N].to(dev)]
        log(f"# labels of the privates {y_raw.tolist()}  (wrong-release {y_other.tolist()}); head NOT extended, m={m}")

    # ---- the chart. eps definition copied verbatim from cifar_charts.py so the ladder is the same construction.
    mean = Pub.mean(0); _, S_, Vh_ = torch.linalg.svd(Pub - mean, full_matrices=False); V_pca = Vh_[: a.k].T.contiguous()
    if is_oracle:
        go = torch.Generator().manual_seed(a.seed + 99)
        Xd = X_raw - mean[:, None]
        noise = torch.randn(Xd.shape, generator=go, dtype=torch.float64).to(dev)
        noise = noise / noise.norm(dim=0, keepdim=True) * Xd.norm(dim=0, keepdim=True) * eps
        Q_, _ = torch.linalg.qr(torch.cat([Xd + noise, V_pca[:, : max(a.k - a.N, 0)]], 1))
        V = Q_[:, : a.k].contiguous()
    else:
        V = V_pca
    psi = lambda Z: mean[:, None] + V @ Z; coords = lambda X: V.T @ (X - mean[:, None])
    X_proj = psi(coords(X_raw))
    proj_err = (torch.linalg.norm(X_proj - X_raw, dim=0) / torch.linalg.norm(X_raw, dim=0))   # THE quantity for the axis
    coord_std = coords(Pub[:5000].T).std(dim=1, keepdim=True)
    log(f"# {a.example} / {a.chart}{' / WRONG-RELEASE CONTROL' if a.wrong_release else ''}  backbone {te*100:.1f}% test"
        f"  |  eps={eps}  measured projection error of the true images: mean {float(proj_err.mean()):.4f}, "
        f"range {float(proj_err.min()):.4f}–{float(proj_err.max()):.4f}")
    if is_oracle: log("# NOT ATTACKER-AVAILABLE: this chart is built from the private images themselves.")

    # ---- the release, fine-tuned on the RAW images
    X_train = X_other if a.wrong_release else X_raw
    H = phi(X_train); y = y_other if a.wrong_release else y_raw
    A0 = (1.0 / math.sqrt(n) * torch.randn(a.r, n, generator=torch.Generator().manual_seed(a.seed + 7), dtype=torch.float64)).to(dev)
    A, B = A0.clone(), torch.zeros(m, a.r, dtype=torch.float64, device=dev)
    Y = torch.eye(m, device=dev, dtype=torch.float64)[y].T
    with torch.no_grad(): soft_res_0 = torch.linalg.norm(torch.softmax(W0 @ H, 0) - Y, dim=0)      # what the batch had to learn
    for _ in range(a.T):
        z = W0 @ H + B @ (A @ H); D = (torch.softmax(z, 0) - Y) / a.N
        B, A = B - a.lr * (D @ (A @ H).T), A - a.lr * (B.T @ D @ H.T)
    A_T, B_T = A, B
    with torch.no_grad(): soft_res_T = torch.linalg.norm(torch.softmax(W0 @ H + B_T @ (A_T @ H), 0) - Y, dim=0)
    sB = torch.linalg.svdvals(B_T); Np = int((sB > a.tol * sB[0]).sum())
    log(f"# recording strength: ||B_T||_F {float(torch.linalg.norm(B_T)):.3e}, ||B_T A_T||_F {float(torch.linalg.norm(B_T @ A_T)):.3e}, "
        f"sigma_N/sigma_1 of B_T {float(sB[min(a.N, len(sB)) - 1] / sB[0]):.2e}; per-image softmax residual at W0 "
        f"{[f'{float(v):.2e}' for v in soft_res_0]} -> at T {[f'{float(v):.2e}' for v in soft_res_T]}")
    _, _, VhB = torch.linalg.svd(B_T, full_matrices=False); Q = VhB[:Np].T
    C = A_T - Q @ (Q.T @ A_T)
    with torch.no_grad():
        Hr = phi(X_raw)                                                            # scored against the true PRIVATES always
        res_truth = torch.linalg.norm(C @ Hr, dim=0) / torch.linalg.norm(A_T @ Hr, dim=0)
        feat_ref = float(torch.linalg.norm(A_T @ phi(Pub[:256].T), dim=0).median())
    log(f"# release: rank B_T {Np}, rank C {int(torch.linalg.matrix_rank(C, rtol=1e-10))}, certificate residual at the privates "
        f"max {float(res_truth.max()):.2e}")

    # ---- the search: identical solver, starts and landing criterion in every cell
    def fun(w):
        f = phi(psi(w.reshape(a.k, 1)))
        return (C @ f).reshape(-1) / torch.linalg.norm(A_T @ f)
    gs = torch.Generator().manual_seed(a.seed + 31); t0 = time.time(); runs = []; Ws = []
    for s in range(a.starts):
        w0 = (torch.randn(a.k, 1, generator=gs).to(dev) * coord_std).reshape(-1)
        w, obj, it = lm_cert(fun, w0, a.iters)
        with torch.no_grad():
            x = psi(w.reshape(a.k, 1))[:, 0]
            e = [float(torch.linalg.norm(x - X_raw[:, i]) / torch.linalg.norm(X_raw[:, i])) for i in range(a.N)]
            fr = float(torch.linalg.norm(A_T @ phi(x.reshape(-1, 1))) / feat_ref)
        j = int(np.argmin(e))
        runs.append(dict(objective=obj, nearest=j, err=e[j], landed=bool(e[j] < 1e-2), degenerate=bool(fr < 0.05)))
        Ws.append(w.detach().cpu())
        if (s + 1) % 100 == 0 or s + 1 == a.starts: log(f"   {s+1}/{a.starts} starts, {time.time()-t0:.0f}s, landed {sum(r['landed'] for r in runs)}")
    W = torch.stack(Ws, 1).to(dev); X_found = psi(W)

    with torch.no_grad():
        err = torch.stack([torch.linalg.norm(X_found - X_raw[:, i:i + 1], dim=0) / torch.linalg.norm(X_raw[:, i]) for i in range(a.N)], 1)
        best_idx = err.argmin(0)
        pub_proj = psi(coords(Pub[:200].T))
        ctrl = pub_proj[:, torch.cdist(X_raw.T, pub_proj.T).argmin(1)]              # nearest public image, through the same chart
        order = sorted(range(len(runs)), key=lambda s: (runs[s]["objective"] if not runs[s]["degenerate"] else float("inf")))
    per = []
    for i in range(a.N):
        bi = int(best_idx[i]); xb = X_found[:, bi]
        per.append(dict(i=i, best_err=float(err[bi, i]), landed=bool(err[bi, i] < 1e-2), best_objective=runs[bi]["objective"],
                        ssim_attack=ssim(xb, X_raw[:, i], shape), ssim_ceiling=ssim(X_proj[:, i], X_raw[:, i], shape),
                        ssim_control=max(ssim(X_found[:, s], ctrl[:, i], shape) for s in range(0, a.starts, max(1, a.starts // 80))),
                        proj_err=float(proj_err[i])))
    mean_ = lambda v: sum(v) / len(v)
    row = dict(part="oracle_ladder", example=a.example, dataset=ex["dataset"], arch=ex["arch"], class_name=cname, shape=list(shape),
               chart=("pca" if not is_oracle else "oracle"),
               eps=(None if not is_oracle else eps), attacker_available=(not is_oracle), wrong_release=a.wrong_release,
               N=a.N, r=a.r, k=a.k, T=a.T, lr=a.lr, seed=a.seed, starts=a.starts,
               backbone_ckpt=bbrec["ckpt"], backbone_test_acc=te, backbone_train_acc=tr, backbone_train_loss=bbrec["train_loss"],
               wp0_fully_trained=wp0, base_gate_note=ex.get("base_gate_note"), private_join_idx=join_idx, wrong_release_idx=other_idx,
               m=m, head_extended=(ylab is None), labels=[int(v) for v in y], labels_of_privates=[int(v) for v in y_raw],
               n_prime=Np, rank_B_T=Np, rank_C=int(torch.linalg.matrix_rank(C, rtol=1e-10)),
               B_T_fro=float(torch.linalg.norm(B_T)), BA_T_fro=float(torch.linalg.norm(B_T @ A_T)),
               B_T_spectrum_rel=[float(v / sB[0]) for v in sB], B_T_sigma_ratio=float(sB[min(a.N, len(sB)) - 1] / sB[0]),
               softmax_residual_at_W0=[float(v) for v in soft_res_0], softmax_residual_at_T=[float(v) for v in soft_res_T],
               proj_err_mean=float(proj_err.mean()), proj_err_min=float(proj_err.min()), proj_err_max=float(proj_err.max()),
               proj_err_per_image=[float(v) for v in proj_err],
               residual_at_truths_max=float(res_truth.max()), residual_at_truths=[float(v) for v in res_truth],
               objective_median=float(torch.tensor([r["objective"] for r in runs]).median()),
               objective_min=float(min(r["objective"] for r in runs if not r["degenerate"])),
               landed=sum(r["landed"] for r in runs), images_found=int(sum(1 for p in per if p["landed"])),
               top20_all_landings=bool(all(runs[s]["landed"] for s in order[:20])),
               top20_landed_count=sum(1 for s in order[:20] if runs[s]["landed"]),
               ssim_attack_mean=mean_([p["ssim_attack"] for p in per]), ssim_ceiling_mean=mean_([p["ssim_ceiling"] for p in per]),
               ssim_control_mean=mean_([p["ssim_control"] for p in per]), per_image=per,
               n_degenerate=sum(r["degenerate"] for r in runs), seconds=time.time() - t0,
               host=socket.gethostname(), cmd=" ".join(sys.argv))
    tag = f"{a.example}_{'pca' if not is_oracle else f'eps{eps:g}'}{'_wrongrelease' if a.wrong_release else ''}"
    with open(os.path.join(a.out_dir, "rows.jsonl"), "a") as f: f.write(json.dumps(row) + "\n")
    torch.save(dict(x_raw=X_raw.cpu(), x_proj=X_proj.cpu(), x_found_best=X_found[:, best_idx].cpu(), W=W.cpu(), runs=runs,
                    A_T=A_T.cpu(), B_T=B_T.cpu(), C=C.cpu(), chart_mean=mean.cpu(), chart_V=V.cpu(), row=row),
               os.path.join(a.out_dir, f"{tag}.pth"))
    log(f"\n=== {tag}: landed {row['landed']}/{a.starts}, images {row['images_found']}/{a.N}, "
        f"top-20 all landings {row['top20_all_landings']}, SSIM attack/ceiling/control "
        f"{row['ssim_attack_mean']:.2f}/{row['ssim_ceiling_mean']:.2f}/{row['ssim_control_mean']:.2f}, "
        f"measured projection error {row['proj_err_mean']:.4f}")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
