"""Train a WIDE+DEEP MLP base model (theta_0) on CIFAR-10 binary-parity — the 'monster network' test.

The reconstruction/bridge testbed has only ever used the 784-1000-1000-1 MLP. This trains a much bigger
net (e.g. 3072-2048-2048-2048-2048-1, ~19M params) on a suiting rich dataset (CIFAR-10, 32x32 RGB) so we
can ask whether the attack survives scale. CIFAR parity (class % 2) matches experiments.LABELS_DICT, so
the downstream bridge needs no label changes. Saves {'state_dict': ...} loadable by load_pretrained.

Small init (1e-4) + full-batch GD on a small memorized set = the Haim max-margin recipe. For the T=1
bridge, near-zero training loss + a positive margin is the operative target.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

import torch
import torch.nn.functional as F

from experiments.run_experiment_b import _build_network
from experiments.configs import DATASETS_DIR, MODELS_DIR


def load_cifar_binary(n_per_class, train=True, device='cuda', seed=0):
    """Balanced CIFAR-10 parity set: class%2 label (matches experiments.LABELS_DICT)."""
    import torchvision
    tfm = torchvision.transforms.ToTensor()
    ds = torchvision.datasets.CIFAR10(DATASETS_DIR, train=train, transform=tfm, download=True)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ds), generator=g).tolist()
    xs, ys, cnt = [], [], {0: 0, 1: 0}
    for j in order:
        img, cls = ds[j]
        lbl = cls % 2
        if cnt[lbl] < n_per_class:
            xs.append(img); ys.append(lbl); cnt[lbl] += 1
        if cnt[0] >= n_per_class and cnt[1] >= n_per_class:
            break
    x = torch.stack(xs).to(device).to(torch.get_default_dtype())
    y = torch.tensor(ys, device=device, dtype=torch.get_default_dtype())
    return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hidden', type=int, nargs='+', default=[2048, 2048, 2048, 2048])
    p.add_argument('--dpc', type=int, default=250)          # data per class -> N=500 base train set
    p.add_argument('--epochs', type=int, default=60000)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--init_scale', type=float, default=1e-4)
    p.add_argument('--out', default=os.path.join(MODELS_DIR, 'weights-cifar10_monster.pth'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    x, y = load_cifar_binary(args.dpc, train=True, device=args.device)
    print(f"CIFAR-10 parity base set: {tuple(x.shape)}  balance {int((y==0).sum())}/{int((y==1).sum())}")
    model = _build_network(device=args.device, input_dim=3072, hidden=args.hidden)
    print(f"monster arch: 3072 -> {args.hidden} -> 1  "
          f"({sum(w.numel() for w in model.parameters())/1e6:.1f}M params)")

    # Variance-preserving Kaiming init. PyTorch's default Linear init undershoots the ReLU-preserving
    # scale by ~2.4x/layer, so a 5-layer forward COLLAPSES (logit std 0.002 -> stuck at ln2). Kaiming
    # (relu gain) restores O(1) logits (std ~0.26). Optional <1 multiplier for a small-init flavor.
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if args.init_scale != 1.0:
                    m.weight.mul_(args.init_scale)
                if m.bias is not None:
                    m.bias.zero_()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sgn = 2 * y - 1
    for ep in range(args.epochs):
        opt.zero_grad()
        logits = model(x).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward(); opt.step()
        if ep % 2000 == 0 or ep == args.epochs - 1:
            with torch.no_grad():
                acc = ((logits > 0).double() == y).double().mean().item()
                mm = (logits * sgn).min().item()
            print(f"ep {ep:>6d}: loss {loss.item():.3e}  train-acc {acc:.3f}  min-margin {mm:+.3f}",
                  flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'epoch': args.epochs, 'batch': None,
                'hidden': args.hidden}, args.out)
    print(f"saved monster theta_0 -> {args.out}")


if __name__ == '__main__':
    main()
