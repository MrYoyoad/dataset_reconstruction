#!/usr/bin/env python3
"""Train a DEEP frozen MNIST MLP so the pixel-space constraint count can be measured past three layers.

The 3-point curve (58 -> 102 -> 158 independent conditions on pixels at 1, 2, 3 adapted layers) is an
extrapolation the moment it is read as a trend, and a three-point extrapolation must not reach a document.
This trains 784 -> width x (D-1) -> 10, GELU, bias on the first layer only -- the same family as the repo's
784-1000-1000-10 backbone so the 3-layer numbers stay on the same axis -- at whatever depth the curve needs.

Plain (no residual, no norm) by default, because a residual path changes the pixel-space Jacobian's rank for
reasons that have nothing to do with the certificate; --residual is there only if plain fails to train.

  python -u -m experiments.exact_inversion.train_deep_backbone --depth 15 --width 1000
"""
import argparse, json, os, socket, time
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
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device(a.device); torch.manual_seed(a.seed)
    Xtr, ytr = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr, device=dev).float(); ytr_t = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev).float(); yte_t = torch.tensor(yte, device=dev)

    dims = [(784, a.width)] + [(a.width, a.width)] * (a.depth - 2) + [(a.width, 10)]
    lins = nn.ModuleList([nn.Linear(i, o, bias=(n == 0)) for n, (i, o) in enumerate(dims)]).to(dev).float()

    def fwd(x):
        h = x
        for n, lin in enumerate(lins):
            z = lin(h)
            if n and a.residual and z.shape == h.shape: z = z + h
            h = torch.nn.functional.gelu(z) if n < len(lins) - 1 else z
        return h

    opt = torch.optim.Adam(lins.parameters(), lr=a.lr); lossf = nn.CrossEntropyLoss()
    print(f"# deep backbone 784-{a.width}x{a.depth-1}-10 GELU residual={a.residual} "
          f"depth={a.depth} git={git_hash()} host={socket.gethostname()}", flush=True)
    best = 0.0
    for ep in range(a.epochs):
        perm = torch.randperm(Xtr_t.shape[0], device=dev)
        for i in range(0, Xtr_t.shape[0], a.bs):
            j = perm[i:i + a.bs]
            opt.zero_grad(); lossf(fwd(Xtr_t[j]), ytr_t[j]).backward(); opt.step()
        with torch.no_grad():
            acc = float((fwd(Xte_t).argmax(1) == yte_t).double().mean())
        best = max(best, acc)
        print(f"  epoch {ep+1:3d}  test acc {acc*100:.2f}%", flush=True)
    stem = f"mnist_mlp_d{a.depth}w{a.width}" + ("_res" if a.residual else "")
    path = os.path.join(a.out_dir, stem + ".pth")
    Ws = [l.weight for l in lins]                           # (out, in): deep_stack applies Ws[l] @ h with h (in, N)
    save_deep(path, Ws, lins[0].bias, acc, dict(depth=a.depth, width=a.width, residual=a.residual,
                                                git=git_hash(), epochs=a.epochs, seed=a.seed))
    flag = "OK" if acc >= a.min_acc else "WEAK -- the curve on this checkpoint is a curve on a bad encoder"
    print(json.dumps(dict(part="DEEP_BACKBONE", path=path, depth=a.depth, width=a.width, residual=a.residual,
                          test_acc=acc, best_acc=best, status=flag, git=git_hash())), flush=True)
    print(f"# saved {path}  test acc {acc*100:.2f}%  [{flag}]", flush=True)
    raise SystemExit(0 if acc >= a.min_acc else 3)      # 3 = trained but WEAK; the runner retries with --residual


if __name__ == "__main__":
    main()
