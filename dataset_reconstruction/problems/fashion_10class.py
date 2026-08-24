"""FashionMNIST 10-class problem (Tier B). Identical structure to mnist_10class
but on FashionMNIST. CrossEntropy training (Main.get_loss_ce branches on
args.output_dim>1)."""
import torch
import torchvision.datasets
import torchvision.transforms

NUM_CLASSES = 10


def load_bound_dataset(dataset, batch_size, shuffle=False, start=None, end=None, **kwargs):
    def _bound(ds, s, e):
        s = 0 if s is None else s
        e = len(ds) if e is None else e
        return torch.utils.data.Subset(ds, range(s, e))
    dataset = _bound(dataset, start, end)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, **kwargs)


def fetch_fashion(root, train=False, transform=None, target_transform=None):
    transform = transform if transform is not None else torchvision.transforms.ToTensor()
    return torchvision.datasets.FashionMNIST(root, train=train, transform=transform,
                                             target_transform=target_transform, download=True)


def load_fashion(root, batch_size, train=False, transform=None, target_transform=None, **kwargs):
    dataset = fetch_fashion(root, train, transform, target_transform)
    return load_bound_dataset(dataset, batch_size, **kwargs)


def move_to_type_device(x, y, device):
    x = x.to(torch.get_default_dtype())
    y = y.to(torch.get_default_dtype())   # get_loss_ce casts to long for CE
    return x.to(device), y.to(device)


def get_balanced_data(args, data_loader, data_amount):
    print('BALANCING DATASET (fashion 10-class)...')
    per_class = data_amount // NUM_CLASSES
    counter = {c: 0 for c in range(NUM_CLASSES)}
    x0, y0 = [], []
    for bx, by in data_loader:
        for i in range(len(bx)):
            c = int(by[i])
            if counter[c] < per_class:
                counter[c] += 1
                x0.append(bx[i])
                y0.append(int(by[i]))          # identity label (the true class)
        if all(counter[c] >= per_class for c in counter):
            break
    x0 = torch.stack(x0)
    y0 = torch.tensor(y0)
    return x0, y0


def load_fashion_data(args):
    data_loader = load_fashion(root=args.datasets_dir, batch_size=100, train=True,
                               shuffle=False, start=0, end=50000)
    x0, y0 = get_balanced_data(args, data_loader, args.data_amount)

    print('LOADING TESTSET')
    data_loader = load_fashion(root=args.datasets_dir, batch_size=100, train=False,
                               shuffle=False, start=0, end=10000)
    x0_test, y0_test = get_balanced_data(args, data_loader, args.data_test_amount)

    x0, y0 = move_to_type_device(x0, y0, args.device)
    x0_test, y0_test = move_to_type_device(x0_test, y0_test, args.device)
    print(f'BALANCE (fashion 10-class): ' + ', '.join(f'{c}:{int((y0 == c).sum())}'
                                                     for c in range(NUM_CLASSES)))
    return [(x0, y0)], [(x0_test, y0_test)], None


def get_dataloader(args):
    args.input_dim = 28 * 28
    args.num_classes = NUM_CLASSES
    args.output_dim = NUM_CLASSES
    args.dataset = 'fashion'

    if args.run_mode == 'reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000
    return load_fashion_data(args)
