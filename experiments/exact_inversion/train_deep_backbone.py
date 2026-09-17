#!/usr/bin/env python3
"""Train a DEEP frozen MNIST MLP so the pixel-space constraint count can be measured past three layers.

The 3-point curve (58 -> 102 -> 158 independent conditions on pixels at 1, 2, 3 adapted layers) is an
extrapolation the moment it is read as a trend, and a three-point extrapolation must not reach a document.
This trains 784 -> width x (D-1) -> 10, GELU, bias on the first layer only -- the same family as the repo's
784-1000-1000-10 backbone so the 3-layer numbers stay on the same axis -- at whatever depth the curve needs.

Plain (no residual, no norm) by default, because a residual path changes the pixel-space Jacobian's rank for
reasons that have nothing to do with the certificate; --residual is there only if plain fails to train.

  python -u -m experiments.exact_inversion.train_deep_backbone --depth 15 --width 1000

WP0 (2026-09-18) "fully trained" mode, the same flags as train_strong_backbone.py: --init-from CKPT (warm-start
from a deep_stack checkpoint: a continuation of the original's training), --target-train-acc A / --min-train-loss L
(stop at the first epoch whose FULL-train accuracy >= A and mean cross-entropy <= L), --max-epochs E, --out PATH
(exact output path; never overwritten).  Adam, no augmentation, no weight decay; lr halves after 5 epochs without
train-loss improvement (floor 1e-5).  Train acc / loss / margin fraction are recorded in the checkpoint dict.

  python -u -m experiments.exact_inversion.train_deep_backbone --init-from models/exact_inversion/mnist_mlp_d15w1000.pth \
      --out models/exact_inversion/mnist_mlp_d15w1000_full.pth --target-train-acc 0.995 --min-train-loss 1e-2 --max-epochs 300
"""
import argparse, json, os, socket, sys, time
import torch, torch.nn as nn

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import read_idx
from experiments.exact_inversion.deep_stack import save_deep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=15, help="number of weight matrices = number of adaptable layers")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--residual", action="store_true", help="fallback only; changes the pixel Jacobian's rank")
    ap.add_argument("--min-acc", type=float, default=0.90, help="below this the checkpoint is flagged WEAK")
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--out-dir", default="models/exact_inversion")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--init-from", default=None, help="warm-start from a deep_stack checkpoint (depth/width read from it)")
    ap.add_argument("--target-train-acc", type=float, default=None); ap.add_argument("--min-train-loss", type=float, default=None)
    ap.add_argument("--max-epochs", type=int, default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--plateau-patience", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.out and os.path.exists(a.out): raise SystemExit(f"refusing to overwrite {a.out}")
    init = torch.load(a.init_from, map_location="cpu", weights_only=False) if a.init_from else None
    if init is not None:
        a.depth, a.width, a.residual = len(init["Ws"]), init["Ws"][0].shape[0], bool(init.get("residual", False))
    dev = torch.device(a.device); torch.manual_seed(a.seed)
    Xtr, ytr = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr, device=dev).float(); ytr_t = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev).float(); yte_t = torch.tensor(yte, device=dev)

    dims = [(784, a.width)] + [(a.width, a.width)] * (a.depth - 2) + [(a.width, 10)]
    lins = nn.ModuleList([nn.Linear(i, o, bias=(n == 0)) for n, (i, o) in enumerate(dims)]).to(dev).float()
    if init is not None:
        with torch.no_grad():
            for lin, w in zip(lins, init["Ws"]): lin.weight.copy_(w.float())
            lins[0].bias.copy_(init["b1"].float())

    def fwd(x):
        h = x
        for n, lin in enumerate(lins):
            z = lin(h)
            if n and a.residual and z.shape == h.shape: z = z + h
            h = torch.nn.functional.gelu(z) if n < len(lins) - 1 else z
        return h

    @torch.no_grad()
    def stats(X, y, chunk=5000):
        correct = 0; loss = 0.0; pos = 0
        for i in range(0, X.shape[0], chunk):
            z = fwd(X[i:i + chunk]); yb = y[i:i + chunk]
            loss += float(lossf(z, yb) * len(yb)); correct += int((z.argmax(1) == yb).sum())
            zy = z.gather(1, yb[:, None])[:, 0]; zo = z.clone(); zo.scatter_(1, yb[:, None], -float("inf"))
            pos += int((zy - zo.max(1).values > 0).sum())
        return dict(acc=correct / X.shape[0], loss=loss / X.shape[0], margin_pos_frac=pos / X.shape[0])

    opt = torch.optim.Adam(lins.parameters(), lr=a.lr); lossf = nn.CrossEntropyLoss()
    rule = a.target_train_acc is not None or a.min_train_loss is not None
    n_epochs = (a.max_epochs or a.epochs) if rule else a.epochs
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=a.plateau_patience, min_lr=1e-5)
             if rule else None)
    print(f"# deep backbone 784-{a.width}x{a.depth-1}-10 GELU residual={a.residual} "
          f"depth={a.depth} git={git_hash()} host={socket.gethostname()}" + (f"  warm-start {a.init_from}" if init else "") +
          (f"  RULE train_acc>={a.target_train_acc} & train_loss<={a.min_train_loss}, max {n_epochs} epochs, Adam lr {a.lr}, "
           f"no augmentation, no weight decay" if rule else ""), flush=True)
    best = 0.0; stopped_at = None; t0 = time.time()
    if rule:
        tr = stats(Xtr_t, ytr_t); print(f"  epoch   0  train acc {tr['acc']*100:.3f}%  train loss {tr['loss']:.3e}", flush=True)
    for ep in range(n_epochs):
        perm = torch.randperm(Xtr_t.shape[0], device=dev)
        for i in range(0, Xtr_t.shape[0], a.bs):
            j = perm[i:i + a.bs]
            opt.zero_grad(); lossf(fwd(Xtr_t[j]), ytr_t[j]).backward(); opt.step()
        with torch.no_grad():
            acc = float((fwd(Xte_t).argmax(1) == yte_t).double().mean())
        best = max(best, acc)
        if rule:
            tr = stats(Xtr_t, ytr_t); sched.step(tr["loss"])
            print(f"  epoch {ep+1:3d}  train acc {tr['acc']*100:.3f}%  train loss {tr['loss']:.3e}  test acc {acc*100:.2f}%  "
                  f"lr {opt.param_groups[0]['lr']:.1e}  {time.time()-t0:.0f}s", flush=True)
            if ((a.target_train_acc is None or tr["acc"] >= a.target_train_acc) and
                    (a.min_train_loss is None or tr["loss"] <= a.min_train_loss)):
                stopped_at = ep + 1; print(f"  stopping rule met at epoch {stopped_at}", flush=True); break
        else:
            print(f"  epoch {ep+1:3d}  test acc {acc*100:.2f}%", flush=True)
    stem = f"mnist_mlp_d{a.depth}w{a.width}" + ("_res" if a.residual else "")
    path = a.out or os.path.join(a.out_dir, stem + ".pth")
    tr = stats(Xtr_t, ytr_t); te = stats(Xte_t, yte_t); acc = te["acc"]; epochs_run = stopped_at or n_epochs
    Ws = [l.weight for l in lins]                           # (out, in): deep_stack applies Ws[l] @ h with h (in, N)
    save_deep(path, Ws, lins[0].bias, acc, dict(depth=a.depth, width=a.width, residual=a.residual,
                                                git=git_hash(), epochs=epochs_run, seed=a.seed,
                                                train_acc=tr["acc"], train_loss=tr["loss"],
                                                train_margin_pos_frac=tr["margin_pos_frac"], test_loss=te["loss"],
                                                init_from=a.init_from, lr=a.lr, bs=a.bs, optimizer="Adam",
                                                augmentation=False, weight_decay=0.0,
                                                stopping_rule=(dict(target_train_acc=a.target_train_acc,
                                                                    min_train_loss=a.min_train_loss, max_epochs=n_epochs,
                                                                    met=stopped_at is not None,
                                                                    plateau_patience=a.plateau_patience,
                                                                    lr_final=opt.param_groups[0]["lr"]) if rule else None),
                                                host=socket.gethostname(), cmd=" ".join(sys.argv)))
    flag = "OK" if acc >= a.min_acc else "WEAK -- the curve on this checkpoint is a curve on a bad encoder"
    print(json.dumps(dict(part="DEEP_BACKBONE", path=path, depth=a.depth, width=a.width, residual=a.residual,
                          test_acc=acc, best_acc=best, train_acc=tr["acc"], train_loss=tr["loss"],
                          epochs=epochs_run, rule_met=(stopped_at is not None) if rule else None,
                          status=flag, git=git_hash())), flush=True)
    print(f"# saved {path}  test acc {acc*100:.2f}%  [{flag}]", flush=True)
    raise SystemExit(0 if acc >= a.min_acc else 3)      # 3 = trained but WEAK; the runner retries with --residual


if __name__ == "__main__":
    main()
