#!/usr/bin/env python3
"""Train MNIST backbones of the SAME architecture to different accuracies, to isolate encoder QUALITY.

The trained-backbone result currently rests on the repo's checkpoint, which reaches 78.45% because it was
trained on a small balanced subset for the reconstruction setting.  On it the conditioning penalty against
a random encoder grew from 7x at k=6 to 1100x at k=17.  The open question -- and it is the load-bearing one,
since every realistic release sits on a strong backbone -- is whether a GOOD encoder makes that worse,
better, or leaves it unchanged.

Same architecture as the repo's model so the only variable is quality: 784 -> 1000 -> 1000 -> 10, GELU,
bias on the first layer only (CreateModel.NeuralNetwork with use_bias=False).  Trained on the FULL train
split.  Checkpoints are written in the repo's own format (a dict with a "state_dict" key) so
trained_backbone.py loads them unchanged.

Saves a mid-accuracy checkpoint (first to cross --mid-acc) and a final one, giving three points on the
quality axis together with the existing 78.45% model: weak / mid / strong.

  python -m experiments.exact_inversion.train_strong_backbone --out-dir models/exact_inversion
"""
import argparse, json, os, socket, sys, time
import numpy as np
import torch, torch.nn as nn

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import read_idx


def build():
    return nn.ModuleList([nn.Linear(784, 1000), nn.Linear(1000, 1000, bias=False),
                          nn.Linear(1000, 10, bias=False)])


def fwd(layers, x):
    h = torch.nn.functional.gelu(layers[0](x))
    h = torch.nn.functional.gelu(layers[1](h))
    return layers[2](h)


def save_ckpt(layers, path, acc, epoch):
    sd = {"layers.0.weight": layers[0].weight.detach().double().cpu(),
          "layers.0.bias":   layers[0].bias.detach().double().cpu(),
          "layers.1.weight": layers[1].weight.detach().double().cpu(),
          "layers.2.weight": layers[2].weight.detach().double().cpu()}
    torch.save({"state_dict": sd, "epoch": epoch, "batch": None, "test_acc": acc}, path)
    print(f"    saved {path}  (test acc {acc*100:.2f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mid-acc", type=float, default=0.90)
    ap.add_argument("--target-acc", type=float, default=0.97)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--out-dir", default="models/exact_inversion")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    Xtr, ytr = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr, device=dev).float(); ytr_t = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev).float(); yte_t = torch.tensor(yte, device=dev)
    layers = build().to(dev).float()
    opt = torch.optim.Adam(layers.parameters(), lr=a.lr)
    lossf = nn.CrossEntropyLoss()
    mid_done = False
    print(f"# training 784-1000-1000-10 GELU on the FULL train split ({Xtr_t.shape[0]} images)  "
          f"git={git_hash()} host={socket.gethostname()}", flush=True)
    for ep in range(a.epochs):
        perm = torch.randperm(Xtr_t.shape[0], device=dev)
        for i in range(0, Xtr_t.shape[0], a.bs):
            j = perm[i:i + a.bs]
            opt.zero_grad(); lossf(fwd(layers, Xtr_t[j]), ytr_t[j]).backward(); opt.step()
        with torch.no_grad():
            acc = float((fwd(layers, Xte_t).argmax(1) == yte_t).float().mean())
        print(f"  epoch {ep+1:>2}  test acc {acc*100:.2f}%", flush=True)
        if not mid_done and acc >= a.mid_acc:
            save_ckpt(layers, os.path.join(a.out_dir, "mnist_mlp_mid.pth"), acc, ep + 1); mid_done = True
        if acc >= a.target_acc and ep >= 4:
            save_ckpt(layers, os.path.join(a.out_dir, "mnist_mlp_strong.pth"), acc, ep + 1)
    with torch.no_grad():
        acc = float((fwd(layers, Xte_t).argmax(1) == yte_t).float().mean())
    save_ckpt(layers, os.path.join(a.out_dir, "mnist_mlp_strong.pth"), acc, a.epochs)
    print(json.dumps(dict(final_test_acc=acc, epochs=a.epochs, git=git_hash(),
                          host=socket.gethostname(), cmd=" ".join(sys.argv))), flush=True)


if __name__ == "__main__":
    main()
