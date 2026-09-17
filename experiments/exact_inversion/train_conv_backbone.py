#!/usr/bin/env python3
"""Train (or continue) a `conv_certificate` MNIST conv backbone to the WP0 base-training gate.

`conv_certificate.py` trains its backbone inline for a fixed 6 epochs and records only the test accuracy, so
`mnist_conv_deep.pth` (98.6% test) has no train-side numbers.  This script reuses `conv_certificate.train_backbone`
-- same SPECS, same module structure, Adam, no augmentation, no weight decay -- with the stopping rule
train_acc >= --target-train-acc AND train_loss <= --min-train-loss (cap --max-epochs), optionally warm-started from
an existing checkpoint (--init-from), and writes the SAME checkpoint format (Wms / bs / Whead / bhead / test_acc /
git) plus the train-side numbers, so `conv_certificate.py --ckpt` and every WP1 harness load it unchanged.
Never overwrites: the output path must not exist.

  python -u -m experiments.exact_inversion.train_conv_backbone --spec deep \
      --init-from models/exact_inversion/mnist_conv_deep.pth --out models/exact_inversion/mnist_conv_deep_full.pth \
      --target-train-acc 0.995 --min-train-loss 1e-2 --max-epochs 300
"""
import argparse, json, os, socket, sys, time
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import read_idx
from experiments.exact_inversion import conv_certificate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="deep", choices=sorted(conv_certificate.SPECS))
    ap.add_argument("--init-from", default=None, help="warm-start from a checkpoint of this format")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-train-acc", type=float, default=0.995)
    ap.add_argument("--min-train-loss", type=float, default=1e-2)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--plateau-patience", type=int, default=5)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device)
    if os.path.exists(a.out): raise SystemExit(f"refusing to overwrite {a.out}")
    conv_certificate.SPEC = conv_certificate.SPECS[a.spec]
    Xtr, ytr = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr, device=dev).float().reshape(-1, 1, 28, 28); ytr_t = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev).float().reshape(-1, 1, 28, 28); yte_t = torch.tensor(yte, device=dev)
    init = torch.load(a.init_from, map_location="cpu", weights_only=False) if a.init_from else None
    print(f"# conv backbone spec '{a.spec}' {conv_certificate.SPEC} dense_hidden={conv_certificate.DENSE_HIDDEN.get(a.spec)}  warm-start {a.init_from}  rule train_acc>={a.target_train_acc} "
          f"& train_loss<={a.min_train_loss}, max {a.max_epochs} epochs, Adam lr {a.lr}, bs {a.bs}, no augmentation, no weight decay  "
          f"git={git_hash()} host={socket.gethostname()}", flush=True)
    t0 = time.time()
    dh = conv_certificate.DENSE_HIDDEN.get(a.spec)
    Wms, bs_, Whead, bhead, acc, st = conv_certificate.train_backbone(
        Xtr_t, ytr_t, Xte_t, yte_t, dev, a.max_epochs, a.bs, a.lr, a.seed, init=init,
        target_train_acc=a.target_train_acc, min_train_loss=a.min_train_loss, plateau_patience=a.plateau_patience,
        return_stats=True, dense_hidden=dh)
    Wd, bd = st.pop("Wd"), st.pop("bd")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(dict(Wms=[w.cpu() for w in Wms], bs=[b.cpu() for b in bs_], Whead=Whead.cpu(), bhead=bhead.cpu(),
                    **({"Wd": Wd.cpu(), "bd": bd.cpu(), "dense_hidden": dh} if Wd is not None else {}),
                    test_acc=acc, git=git_hash(), spec=a.spec, init_from=a.init_from, seed=a.seed, lr=a.lr, batch_size=a.bs,
                    optimizer="Adam", augmentation=False, weight_decay=0.0,
                    stopping_rule=dict(target_train_acc=a.target_train_acc, min_train_loss=a.min_train_loss,
                                       max_epochs=a.max_epochs, met=st["rule_met"], plateau_patience=a.plateau_patience,
                                       lr_final=st["lr_final"]),
                    train_acc=st["train_acc"], train_loss=st["train_loss"], train_margin_pos_frac=st["train_margin_pos_frac"],
                    test_loss=st["test_loss"], epochs_run=st["epochs_run"], seconds=time.time() - t0,
                    host=socket.gethostname(), cmd=" ".join(sys.argv)), a.out)
    print(json.dumps(dict(part="CONV_BACKBONE_FULL", out=a.out, spec=a.spec, **st, seconds=time.time() - t0,
                          git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))), flush=True)
    print(f"# saved {a.out}  train {st['train_acc']*100:.3f}% / loss {st['train_loss']:.3e}  test {acc*100:.2f}%  "
          f"rule {'MET' if st['rule_met'] else 'NOT MET'} after {st['epochs_run']} epochs", flush=True)


if __name__ == "__main__":
    main()
