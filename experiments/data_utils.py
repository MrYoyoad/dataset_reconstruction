"""Few-shot MNIST data loading and control image selection."""

import torch
import torchvision.datasets
import torchvision.transforms
from experiments.configs import DATASETS_DIR, LABELS_DICT


def _load_mnist(train=True, root=None):
    """Load raw MNIST dataset."""
    root = root or DATASETS_DIR
    transform = torchvision.transforms.ToTensor()
    return torchvision.datasets.MNIST(root, train=train, transform=transform, download=True)


def _load_dataset(name='mnist', train=True, root=None):
    """Load MNIST or Fashion-MNIST — both are 28x28x1, so they drop straight into the 784-input MLP.

    Fashion-MNIST is the 'harder data' testbed (Step 4 / Addition 1): unlike MNIST, its dataset mean
    is NOT ~= each image, so leakage numbers (control margin / retrieval) speak for themselves without
    the N=2 background-dominance caveat. Binary labels reuse the parity map (class idx even->0, odd->1)
    via _get_binary_label, giving a balanced 5-vs-5 split.
    """
    root = root or DATASETS_DIR
    transform = torchvision.transforms.ToTensor()
    if name == 'mnist':
        return torchvision.datasets.MNIST(root, train=train, transform=transform, download=True)
    if name == 'fashion':
        return torchvision.datasets.FashionMNIST(root, train=train, transform=transform, download=True)
    if name == 'flowers':
        # Flowers102 (real natural images) run through the SAME 784-MLP cookbook: resize to 28x28 +
        # grayscale so it drops into the MNIST-shaped input. Much harder structure than MNIST; binary
        # label = class-index parity (102 species -> balanced 2-way). split test<->'test', train->'train'.
        flowers_tfm = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
        ])
        return torchvision.datasets.Flowers102(
            root, split=('train' if train else 'test'), transform=flowers_tfm, download=True)
    if name in ('flowers32', 'flowers64'):
        # Flowers-native track: RGB at NATIVE resolution (no grayscale downsample). The base theta_0
        # is trained separately (dataset_reconstruction/problems/flowers102_parity.py) on train+val;
        # here we only load the TEST split (train=False) for fine-tune/control, so the fine-tune data
        # is disjoint from theta_0's training set. hw = 32 (D=3072) or 64 (D=12288).
        hw = 32 if name == 'flowers32' else 64
        flowers_tfm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((hw, hw)),
            torchvision.transforms.ToTensor(),
        ])
        return torchvision.datasets.Flowers102(
            root, split=('train' if train else 'test'), transform=flowers_tfm, download=True)
    if name == 'cifar100':
        # PROXY-only (natural 32x32 RGB, 50k train) for the 'rescue flowers' test: decoder trains on
        # abundant CIFAR-100 through the FLOWERS32 base model (same 3072 geometry). R2F: proxy need not
        # match the private distribution. Binary label = class parity.
        return torchvision.datasets.CIFAR100(root, train=train, transform=transform, download=True)
    if name == 'cifar10':
        # 'Monster network' track: CIFAR-10 at NATIVE 32x32 RGB (D=3072). Binary label = class parity
        # (class%2), which matches LABELS_DICT so _get_binary_label needs no change. theta_0 is the
        # wide+deep monster trained by experiments/train_monster_base.py on the TRAIN split; the victim
        # fine-tune/control here loads the TEST split (disjoint from theta_0's training set).
        return torchvision.datasets.CIFAR10(root, train=train, transform=transform, download=True)
    raise ValueError(
        f"Unknown dataset: {name} (expected 'mnist', 'fashion', 'flowers', 'flowers32', 'flowers64', or 'cifar10')")


def _get_binary_label(digit_label):
    """Binary label. MNIST: odd/even via LABELS_DICT (matching mnist_odd_even.py). Other datasets
    (fashion/flowers, class idx >= 10): fall back to class-index parity — LABELS_DICT[d] == d % 2 for
    MNIST digits, so this is consistent with MNIST and generalizes to any class count."""
    d = int(digit_label)
    return LABELS_DICT[d] if d in LABELS_DICT else d % 2


def get_few_shot_mnist(n_per_class, seed=42, root=None, device='cpu'):
    """Load n_per_class samples per binary class from MNIST train set.

    Uses the same odd/even labeling as problems/mnist_odd_even.py:
    even digits (0,2,4,6,8) → class 0, odd digits (1,3,5,7,9) → class 1.

    Returns:
        x_train: tensor [2*n_per_class, 1, 28, 28], float64
        y_train: tensor [2*n_per_class], float64, values in {0, 1}
        digit_labels: list of int, the actual MNIST digit labels
        indices: list of int, the MNIST dataset indices (for reproducibility)
    """
    dataset = _load_mnist(train=True, root=root)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=rng)

    x_list, y_list, digit_list, idx_list = [], [], [], []
    counts = {0: 0, 1: 0}

    for idx in perm.tolist():
        img, digit = dataset[idx]
        binary_label = _get_binary_label(digit)
        if counts[binary_label] < n_per_class:
            counts[binary_label] += 1
            x_list.append(img)
            y_list.append(binary_label)
            digit_list.append(int(digit))
            idx_list.append(idx)
        if counts[0] >= n_per_class and counts[1] >= n_per_class:
            break

    x_train = torch.stack(x_list).to(torch.float64).to(device)
    y_train = torch.tensor(y_list, dtype=torch.float64, device=device)
    return x_train, y_train, digit_list, idx_list


def get_finetuning_data(n_per_class, seed=42, root=None, device='cpu', dataset='mnist',
                        source='all', holdout_species=None, num_classes=2):
    """Load few-shot fine-tuning data from the TEST set of `dataset`.

    These samples are guaranteed non-overlapping with the pre-trained model's
    training data (which used MNIST train set, first 250/class sequential).

    This is the "private data" in the attack scenario: someone fine-tunes
    a pre-trained model on these samples, and the attacker tries to
    reconstruct them from the weight change.

    Phase-D (Q-B pretrain/finetune overlap): `source` filters which classes/species are eligible,
    relative to `holdout_species` (the set of species HELD OUT of the base model's training):
      - 'all'   (default): no filter — current behavior, byte-identical for MNIST/fashion/flowers.
      - 'seen'  : only species the base model DID train on (species NOT in holdout_species) — the
                  overlap regime (theta_0 already partially fits them).
      - 'novel' : only the held-out species (species IN holdout_species) — the no-overlap regime.
    `holdout_species` is a set/list of raw class indices; required when source != 'all'.

    Returns:
        x_ft: tensor [2*n_per_class, C, H, W], float64
        y_ft: tensor [2*n_per_class], float64, values in {0, 1}
        digit_labels: list of int, the actual class labels
        indices: list of int, the test set indices
    """
    if source != 'all' and holdout_species is None:
        raise ValueError(f"source={source!r} requires holdout_species (the base model's held-out set)")
    holdout = set(int(s) for s in holdout_species) if holdout_species is not None else set()
    dataset = _load_dataset(dataset, train=False, root=root)  # TEST set
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=rng)

    # Tier B: num_classes>2 balances over the K TRUE class labels and returns y as
    # LONG class indices (CrossEntropy requires long — a float y is read as class
    # probabilities, a silent miscompute). Binary path (default) is byte-identical.
    multiclass = num_classes > 2
    x_list, y_list, digit_list, idx_list = [], [], [], []
    counts = {c: 0 for c in range(num_classes)} if multiclass else {0: 0, 1: 0}

    for idx in perm.tolist():
        img, digit = dataset[idx]
        if source == 'seen' and int(digit) in holdout:
            continue          # skip held-out species -> only species theta_0 trained on
        if source == 'novel' and int(digit) not in holdout:
            continue          # skip trained species -> only held-out (novel) species
        label = int(digit) if multiclass else _get_binary_label(digit)
        if label < num_classes and counts[label] < n_per_class:
            counts[label] += 1
            x_list.append(img)
            y_list.append(label)
            digit_list.append(int(digit))
            idx_list.append(idx)
        if all(counts[c] >= n_per_class for c in counts):
            break

    x_ft = torch.stack(x_list).to(torch.float64).to(device)
    if multiclass:
        y_ft = torch.tensor(y_list, dtype=torch.long, device=device)
    else:
        y_ft = torch.tensor(y_list, dtype=torch.float64, device=device)
    return x_ft, y_ft, digit_list, idx_list


def get_control_images_in_distribution(training_digits, seed=99, root=None, device='cpu', dataset='mnist'):
    """Load same-class control images from the test set of `dataset` ('mnist' or 'fashion').

    For each digit in training_digits, finds a different instance of the same
    digit from the test set. This rules out class-prototype explanations.

    Args:
        training_digits: list of int, the actual digit labels used in training
        seed: random seed for selection
        root: MNIST data directory

    Returns:
        x_control: tensor [N, 1, 28, 28], float64
        y_control: tensor [N], float64 (binary labels)
        control_digits: list of int
    """
    dataset = _load_dataset(dataset, train=False, root=root)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=rng)

    # Collect one control per unique digit
    unique_digits = set(training_digits)
    digit_to_img = {}

    for idx in perm.tolist():
        img, digit = dataset[idx]
        digit = int(digit)
        if digit in unique_digits and digit not in digit_to_img:
            digit_to_img[digit] = img
        if len(digit_to_img) == len(unique_digits):
            break

    # Return in the SAME ORDER as training_digits (critical for paired metrics)
    x_list = [digit_to_img[d] for d in training_digits]
    y_list = [_get_binary_label(d) for d in training_digits]

    x_control = torch.stack(x_list).to(torch.float64).to(device)
    y_control = torch.tensor(y_list, dtype=torch.float64, device=device)
    return x_control, y_control, list(training_digits)


def get_control_images_ood(training_digits, seed=99, root=None, device='cpu'):
    """Load same-digit control images from EMNIST-Digits (out-of-distribution).

    EMNIST-Digits has the same digit classes as MNIST but different writers.
    NOTE: EMNIST images are transposed relative to MNIST.

    Args:
        training_digits: list of int, the actual digit labels used in training

    Returns:
        x_control: tensor [N, 1, 28, 28], float64
        y_control: tensor [N], float64 (binary labels)
        control_digits: list of int
    """
    root = root or DATASETS_DIR
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.transpose(1, 2)),  # EMNIST fix
    ])
    dataset = torchvision.datasets.EMNIST(
        root, split='digits', train=False, transform=transform, download=True
    )
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=rng)

    x_list, y_list, digit_list = [], [], []
    needed = {d: 1 for d in training_digits}
    found = {d: 0 for d in training_digits}

    for idx in perm.tolist():
        img, digit = dataset[idx]
        digit = int(digit)
        if digit in needed and found[digit] < needed[digit]:
            found[digit] += 1
            x_list.append(img)
            y_list.append(_get_binary_label(digit))
            digit_list.append(digit)
        if all(found[d] >= needed[d] for d in needed):
            break

    x_control = torch.stack(x_list).to(torch.float64).to(device)
    y_control = torch.tensor(y_list, dtype=torch.float64, device=device)
    return x_control, y_control, digit_list
