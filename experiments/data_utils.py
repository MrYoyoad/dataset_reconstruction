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


def _get_binary_label(digit_label):
    """Map MNIST digit label to binary odd/even label (matching mnist_odd_even.py)."""
    return LABELS_DICT[int(digit_label)]


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


def get_finetuning_data(n_per_class, seed=42, root=None, device='cpu'):
    """Load few-shot fine-tuning data from MNIST TEST set.

    These samples are guaranteed non-overlapping with the pre-trained model's
    training data (which used MNIST train set, first 250/class sequential).

    This is the "private data" in the attack scenario: someone fine-tunes
    a pre-trained model on these samples, and the attacker tries to
    reconstruct them from the weight change.

    Returns:
        x_ft: tensor [2*n_per_class, 1, 28, 28], float64
        y_ft: tensor [2*n_per_class], float64, values in {0, 1}
        digit_labels: list of int, the actual MNIST digit labels
        indices: list of int, the MNIST test set indices
    """
    dataset = _load_mnist(train=False, root=root)  # TEST set
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

    x_ft = torch.stack(x_list).to(torch.float64).to(device)
    y_ft = torch.tensor(y_list, dtype=torch.float64, device=device)
    return x_ft, y_ft, digit_list, idx_list


def get_control_images_in_distribution(training_digits, seed=99, root=None, device='cpu'):
    """Load same-digit control images from MNIST test set.

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
    dataset = _load_mnist(train=False, root=root)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=rng)

    x_list, y_list, digit_list = [], [], []
    needed = {d: 1 for d in training_digits}  # one control per training digit
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
