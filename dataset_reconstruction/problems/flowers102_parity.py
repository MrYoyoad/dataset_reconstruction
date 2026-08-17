"""Flowers102 binary-parity problem for the NATIVE-dimension reconstruction track.

Trains a max-margin MLP base model (theta_0) on Flowers102 at RGB native resolution
(32x32x3 -> D=3072, or 64x64x3 -> D=12288 via --flowers_hw). Binary label = species-index
parity over the 102 species (matches experiments/data_utils._get_binary_label). The base
model pools the train+val splits (~2040 imgs); the disjoint 'test' split supplies the
reconstruction fine-tune/control data downstream (loaded by experiments/data_utils.py, never
here) -> the attack reconstructs data theta_0 never trained on.

Config knobs (GetParams.py): --flowers_hw (32/64), --flowers_gray (RGB by default),
--flowers_holdout (species class-indices HELD OUT of base training, for the Phase-D Q-B
overlap contrast).

Modeled on problems/cifar10_vehicles_animals.py (proven binary-D=3072 recipe).
"""
import torch
import torchvision.datasets
import torchvision.transforms as T


def _flowers_transform(args):
    hw = int(getattr(args, 'flowers_hw', 32))
    gray = bool(getattr(args, 'flowers_gray', False))
    ops = []
    if gray:
        ops.append(T.Grayscale(num_output_channels=1))
    ops += [T.Resize((hw, hw)), T.ToTensor()]
    return T.Compose(ops)


def fetch_flowers(root, train=False, transform=None):
    """train=True -> train+val pooled (theta_0's private training set).
    train=False -> the disjoint 'test' split."""
    transform = transform if transform is not None else T.ToTensor()
    if train:
        tr = torchvision.datasets.Flowers102(root, split='train', transform=transform, download=True)
        va = torchvision.datasets.Flowers102(root, split='val', transform=transform, download=True)
        return torch.utils.data.ConcatDataset([tr, va])
    return torchvision.datasets.Flowers102(root, split='test', transform=transform, download=True)


def move_to_type_device(x, y, device):
    print('X:', x.shape)
    print('y:', y.shape)
    x = x.to(torch.get_default_dtype())
    y = y.to(torch.get_default_dtype())
    return x.to(device), y.to(device)


def create_labels(y0):
    """Binary parity over the 102 species (species idx even -> 0, odd -> 1)."""
    return torch.stack([torch.tensor(int(cur_y) % 2) for cur_y in y0])


def get_balanced_data(args, dataset, data_amount, holdout=None):
    """Collect a parity-balanced set of `data_amount` images, skipping held-out species.

    Iterates deterministically (shuffle=False). holdout = raw species indices excluded from
    theta_0's training (Phase-D Q-B); the loop only breaks once BOTH parity classes are full,
    so ordering never starves a class.
    """
    print('BALANCING DATASET...')
    holdout = set(int(s) for s in (holdout or []))
    per_class = data_amount // 2
    counts = {0: 0, 1: 0}
    xs, ys = [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    done = False
    for bx, by in loader:
        for i in range(len(bx)):
            raw = int(by[i])
            if raw in holdout:
                continue
            lbl = raw % 2
            if counts[lbl] < per_class:
                counts[lbl] += 1
                xs.append(bx[i])
                ys.append(lbl)
            if counts[0] >= per_class and counts[1] >= per_class:
                done = True
                break
        if done:
            break
    if not xs:
        raise RuntimeError('flowers102_parity: no data collected — check split/holdout')
    x0 = torch.stack(xs)
    y0 = torch.tensor(ys)
    print(f'  collected {len(xs)} imgs (class0:{counts[0]} class1:{counts[1]}), '
          f'holdout={sorted(holdout)}')
    return x0, y0


def load_flowers_data(args):
    tfm = _flowers_transform(args)
    holdout = getattr(args, 'flowers_holdout', None)

    print('TRAINSET (train+val pooled) BALANCED')
    train_ds = fetch_flowers(args.datasets_dir, train=True, transform=tfm)
    x0, y0 = get_balanced_data(args, train_ds, args.data_amount, holdout=holdout)

    print('LOADING TESTSET')
    assert not args.data_use_test or (args.data_use_test and args.data_test_amount >= 2), \
        f"args.data_use_test={args.data_use_test} but args.data_test_amount={args.data_test_amount}"
    test_ds = fetch_flowers(args.datasets_dir, train=False, transform=tfm)
    # Test balanced set uses ALL species (holdout is a theta_0-training restriction only; the
    # test split is disjoint from train+val regardless).
    x0_test, y0_test = get_balanced_data(args, test_ds, args.data_test_amount)

    x0, y0 = move_to_type_device(x0, y0, args.device)
    x0_test, y0_test = move_to_type_device(x0_test, y0_test, args.device)
    print(f'BALANCE: 0: {int((y0 == 0).sum())}, 1: {int((y0 == 1).sum())}')
    return [(x0, y0)], [(x0_test, y0_test)], None


def get_dataloader(args):
    hw = int(getattr(args, 'flowers_hw', 32))
    gray = bool(getattr(args, 'flowers_gray', False))
    args.input_dim = hw * hw * (1 if gray else 3)
    args.num_classes = 2
    args.output_dim = 1
    args.dataset = 'flowers102_parity'

    if args.run_mode == 'reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000

    return load_flowers_data(args)
