#!/usr/bin/env python3
"""Train a CIFAR-10 ResNet-18 frozen base TO THE BASE GATE (plan 2026-09-18, package P2(ii); gate rule in
`experiments/exact_inversion/BASE_TRAINING_GATE.md`: train accuracy >= 0.995 AND mean train cross-entropy <= 1e-2 on
the UN-AUGMENTED train split, measured in the job every epoch).

ARCHITECTURE.  torchvision `resnet18(num_classes=10)` adapted to CIFAR resolution: the stem is a 3x3 stride-1 conv
(no 7x7, no maxpool), so the stages sit at 32x32 (layer1, 64 ch) -> 16x16 (layer2, 128 ch) -> 8x8 (layer3, 256 ch)
-> 4x4 (layer4, 512 ch).  BatchNorm everywhere (eval mode = a fixed per-channel affine map, which folds into the
frozen encoder: the certificate pass in `experiments/multilayer_cert/resnet_ranklaw.py` reads it as such).  ReLU
activations, set to `inplace=False` at construction so torch.func (vmap / jvp) can trace the frozen net later; the
map is byte-identical.  Inputs are the raw [0, 1] pixels in CIFAR's (3, 32, 32) layout with NO mean/std
normalisation (as `experiments/cifar/cifar_newclass.py` does) -- the pixel chart of the certificate pass is then
the raw pixel space.  `build_resnet18_cifar()` is the ONE constructor every reader imports (trainer, gate loader,
rank-law harness), so the frozen map can never drift between them.

RECIPE (standard CIFAR ResNet recipe, seed 1): SGD, momentum 0.9, weight decay 5e-4, lr 0.1 with a cosine
schedule over `--epochs` (default 120; after the horizon lr stays at the cosine floor `--lr-min`), batch 128,
augmentation = random crop (4-pixel zero pad) + horizontal flip, done on the GPU on the in-memory tensor.  Every
epoch the net is put in eval mode and the FULL un-augmented train split and the test split are evaluated (FP32);
training stops at the first epoch where the gate holds, or at `--max-epochs` (200) WITHOUT the gate (recorded on the
checkpoint as `rule_met: false` -- the gate is never lowered).  The checkpoint carries `state_dict` (CPU FP32) +
`train_acc, train_loss, test_acc, test_loss, epochs_run, rule_met, git, seed, recipe, ...`.  The independent gate
measurement is `base_training_gate.py --ckpts models/exact_inversion/cifar10_resnet18.pth` (family `cifar_resnet`).

  python -u -m experiments.exact_inversion.train_resnet_backbone --out models/exact_inversion/cifar10_resnet18.pth --seed 1
"""
import argparse, json, math, os, socket, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.cifar.cifar_newclass import load_cifar10

GATE_TRAIN_ACC, GATE_TRAIN_LOSS = 0.995, 1e-2


def build_resnet18_cifar(num_classes=10):
    """torchvision resnet18 with a 3x3 stride-1 stem and no maxpool (CIFAR adaptation); ReLUs not in place."""
    net = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    for m in net.modules():
        if isinstance(m, nn.ReLU): m.inplace = False
    return net


def load_resnet18(path, dev, dtype=torch.float64):
    """Checkpoint -> frozen eval-mode net in `dtype` (FP64 by default: the certificate pass) + the dict."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_resnet18_cifar(ck.get("num_classes", 10))
    net.load_state_dict(ck["state_dict"]); net = net.to(dev).to(dtype).eval()
    for p in net.parameters(): p.requires_grad_(False)
    return net, ck


def augment(x, g):
    """Random crop (pad 4, zero) + horizontal flip on a (B, 3, 32, 32) GPU batch; `g` a torch.Generator on x.device."""
    B = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (B,), generator=g, device=x.device); ox = torch.randint(0, 9, (B,), generator=g, device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (oy[:, None] + ar[None, :])[:, None, :, None]                              # (B,1,32,1)
    cols = (ox[:, None] + ar[None, :])[:, None, None, :]                              # (B,1,1,32)
    bi = torch.arange(B, device=x.device)[:, None, None, None]; ci = torch.arange(3, device=x.device)[None, :, None, None]
    out = xp[bi, ci, rows, cols]
    flip = torch.rand(B, generator=g, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


@torch.no_grad()
def evaluate(net, X, y, bs=1000):
    """Accuracy and mean cross-entropy over a whole split, eval mode, no augmentation."""
    net.eval(); correct = 0; loss = 0.0
    for i in range(0, X.shape[0], bs):
        z = net(X[i:i + bs]); yb = y[i:i + bs]
        loss += float(F.cross_entropy(z, yb, reduction="sum")); correct += int((z.argmax(1) == yb).sum())
    return correct / X.shape[0], loss / X.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120, help="cosine horizon (lr reaches --lr-min here)")
    ap.add_argument("--max-epochs", type=int, default=200, help="hard stop; after the horizon lr stays at --lr-min")
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--lr-min", type=float, default=1e-4); ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=5e-4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target-train-acc", type=float, default=GATE_TRAIN_ACC)
    ap.add_argument("--min-train-loss", type=float, default=GATE_TRAIN_LOSS)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="models/exact_inversion/cifar10_resnet18.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device)
    torch.manual_seed(a.seed); torch.backends.cudnn.benchmark = True
    Xtr, ytr, Xte, yte = load_cifar10(a.data_root)
    Xtr_t = torch.tensor(Xtr, device=dev).reshape(-1, 3, 32, 32); ytr_t = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev).reshape(-1, 3, 32, 32); yte_t = torch.tensor(yte, device=dev)
    net = build_resnet18_cifar().to(dev).float()
    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=a.momentum, weight_decay=a.wd)
    lr_at = lambda ep: a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * min(ep, a.epochs) / a.epochs))
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"# ResNet-18/CIFAR ({n_par/1e6:.2f} M params) on CIFAR-10 {Xtr_t.shape[0]} train; SGD m={a.momentum} wd={a.wd} "
          f"lr {a.lr} cosine->{a.lr_min} over {a.epochs} ep (max {a.max_epochs}), bs {a.bs}, crop+flip; gate train acc >= "
          f"{a.target_train_acc} and CE <= {a.min_train_loss} on the UN-augmented train split; seed {a.seed}  git={git_hash()} "
          f"host={socket.gethostname()} gpu={torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}", flush=True)
    t0 = time.time(); stopped_at = None; hist = []
    for ep in range(a.max_epochs):
        for pg in opt.param_groups: pg["lr"] = lr_at(ep)
        net.train(); perm = torch.randperm(Xtr_t.shape[0], generator=g, device=dev); run_loss = 0.0; nb = 0
        for i in range(0, Xtr_t.shape[0], a.bs):
            j = perm[i:i + a.bs]
            xb = augment(Xtr_t[j], g); yb = ytr_t[j]
            opt.zero_grad(set_to_none=True); loss = F.cross_entropy(net(xb), yb); loss.backward(); opt.step()
            run_loss += float(loss); nb += 1
        tr_acc, tr_loss = evaluate(net, Xtr_t, ytr_t); te_acc, te_loss = evaluate(net, Xte_t, yte_t)
        hist.append(dict(epoch=ep + 1, lr=lr_at(ep), aug_loss=run_loss / nb, train_acc=tr_acc, train_loss=tr_loss,
                         test_acc=te_acc, test_loss=te_loss, seconds=time.time() - t0))
        print(f"  epoch {ep+1:3d}  lr {lr_at(ep):.2e}  aug-batch loss {run_loss/nb:.3e}  UNAUG train acc {tr_acc*100:.3f}% "
              f"CE {tr_loss:.3e}  test acc {te_acc*100:.2f}% CE {te_loss:.3e}  {time.time()-t0:.0f}s", flush=True)
        if tr_acc >= a.target_train_acc and tr_loss <= a.min_train_loss:
            stopped_at = ep + 1; print(f"  gate met at epoch {stopped_at}", flush=True); break
    tr_acc, tr_loss = evaluate(net, Xtr_t, ytr_t); te_acc, te_loss = evaluate(net, Xte_t, yte_t)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ck = dict(state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
              arch="torchvision resnet18, 3x3 s1 stem, no maxpool, BN, ReLU(inplace=False), inputs [0,1] unnormalised",
              num_classes=10, train_acc=tr_acc, train_loss=tr_loss, test_acc=te_acc, test_loss=te_loss,
              epochs_run=stopped_at or a.max_epochs, rule_met=stopped_at is not None,
              stopping_rule=f"train acc >= {a.target_train_acc} and train CE <= {a.min_train_loss} (un-augmented train split)",
              recipe=dict(opt="SGD", lr=a.lr, lr_min=a.lr_min, cosine_epochs=a.epochs, max_epochs=a.max_epochs,
                          momentum=a.momentum, weight_decay=a.wd, bs=a.bs, augmentation="random crop pad4 + hflip",
                          dtype="float32"),
              seed=a.seed, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv),
              seconds=time.time() - t0, history=hist, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    torch.save(ck, a.out)
    print(f"# saved {a.out}: train acc {tr_acc*100:.3f}% CE {tr_loss:.3e}  test acc {te_acc*100:.2f}%  epochs {ck['epochs_run']} "
          f"rule_met={ck['rule_met']}  {time.time()-t0:.0f}s", flush=True)
    print(json.dumps({k: v for k, v in ck.items() if k not in ("state_dict", "history")}), flush=True)
    if not ck["rule_met"]:
        print("# GATE NOT MET within --max-epochs: the checkpoint is saved but is NOT a base for any package", flush=True)


if __name__ == "__main__":
    main()
