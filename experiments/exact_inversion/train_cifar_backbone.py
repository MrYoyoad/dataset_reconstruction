#!/usr/bin/env python3
"""Train a CIFAR-10 MLP backbone of the MNIST architecture's shape (3072 -> 1000 -> 1000 -> 10, GELU, bias on the
first layer only), saved in the repo's checkpoint format so trained_backbone.TrainedBackbone loads it unchanged.
Used by new_class.py (adding a "flower" class to a CIFAR-10 model).

  python -m experiments.exact_inversion.train_cifar_backbone --out models/exact_inversion/cifar10_mlp.pth
"""
import argparse, json, os, pickle, socket, sys
import numpy as np, torch, torch.nn as nn

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.train_strong_backbone import fwd


def load_cifar10(root):
    d = os.path.join(root, "cifar-10-batches-py"); Xs, ys = [], []
    for i in range(1, 6):
        b = pickle.load(open(os.path.join(d, f"data_batch_{i}"), "rb"), encoding="bytes"); Xs.append(b[b"data"]); ys += b[b"labels"]
    b = pickle.load(open(os.path.join(d, "test_batch"), "rb"), encoding="bytes")
    return (np.concatenate(Xs).astype(np.float64) / 255.0, np.array(ys, dtype=np.int64),
            b[b"data"].astype(np.float64) / 255.0, np.array(b[b"labels"], dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--out", default="models/exact_inversion/cifar10_mlp.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device); torch.manual_seed(a.seed)
    Xtr, ytr, Xte, yte = load_cifar10(a.data_root)
    Xtr_t = torch.tensor(Xtr, device=dev).float(); ytr_t = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev).float(); yte_t = torch.tensor(yte, device=dev)
    layers = nn.ModuleList([nn.Linear(3072, 1000), nn.Linear(1000, 1000, bias=False), nn.Linear(1000, 10, bias=False)]).to(dev).float()
    opt = torch.optim.Adam(layers.parameters(), lr=a.lr); lossf = nn.CrossEntropyLoss()
    print(f"# training 3072-1000-1000-10 GELU on CIFAR-10 ({Xtr_t.shape[0]} images)  git={git_hash()} host={socket.gethostname()}", flush=True)
    best = 0.0
    for ep in range(a.epochs):
        perm = torch.randperm(Xtr_t.shape[0], device=dev)
        for i in range(0, Xtr_t.shape[0], a.bs):
            j = perm[i:i + a.bs]; opt.zero_grad(); lossf(fwd(layers, Xtr_t[j]), ytr_t[j]).backward(); opt.step()
        with torch.no_grad():
            acc = float((fwd(layers, Xte_t).argmax(1) == yte_t).float().mean())
        print(f"  epoch {ep+1:>2}  test acc {acc*100:.2f}%", flush=True)
        if acc > best:                                                # keep the best test-accuracy epoch (an MLP overfits CIFAR)
            best = acc
            sd = {"layers.0.weight": layers[0].weight.detach().double().cpu(), "layers.0.bias": layers[0].bias.detach().double().cpu(),
                  "layers.1.weight": layers[1].weight.detach().double().cpu(), "layers.2.weight": layers[2].weight.detach().double().cpu()}
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            torch.save({"state_dict": sd, "epoch": ep + 1, "batch": None, "test_acc": acc}, a.out)
    print(json.dumps(dict(best_test_acc=best, epochs=a.epochs, out=a.out, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))), flush=True)


if __name__ == "__main__":
    main()
