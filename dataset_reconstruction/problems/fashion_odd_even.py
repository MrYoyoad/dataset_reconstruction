"""Fashion-MNIST odd/even parity — a registered training problem (mirrors
mnist_odd_even.py, swapping MNIST -> FashionMNIST). Added for the rigor upgrade:
Fashion needs its OWN honestly-trained base model per activation (not a swap).
Labels = parity of the class index (0=T-shirt even, 1=Trouser odd, ...), same
labels_dict as MNIST. 28x28 grayscale, input_dim=784, binary output."""
import torch
import torchvision.datasets
import torchvision.transforms


def load_bound_dataset(dataset, batch_size, shuffle=False, start=None, end=None, **kwargs):
    def _bound_dataset(dataset, start, end):
        if start is None:
            start = 0
        if end is None:
            end = len(dataset)
        return torch.utils.data.Subset(dataset, range(start, end))

    dataset = _bound_dataset(dataset, start, end)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, **kwargs)


def fetch_fashion(root, train=False, transform=None, target_transform=None):
    transform = transform if transform is not None else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.FashionMNIST(root, train=train, transform=transform,
                                                target_transform=target_transform, download=True)
    return dataset


def load_fashion(root, batch_size, train=False, transform=None, target_transform=None, **kwargs):
    dataset = fetch_fashion(root, train, transform, target_transform)
    return load_bound_dataset(dataset, batch_size, **kwargs)


def move_to_type_device(x, y, device):
    x = x.to(torch.get_default_dtype())
    y = y.to(torch.get_default_dtype())
    x, y = x.to(device), y.to(device)
    return x, y


def create_labels(y0):
    labels_dict = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1}
    y0 = torch.stack([torch.tensor(labels_dict[int(cur_y)]) for cur_y in y0])
    return y0


def get_balanced_data(args, data_loader, data_amount):
    data_amount_per_class = data_amount // 2
    labels_counter = {1: 0, 0: 0}
    x0, y0 = [], []
    got_enough = False
    for bx, by in data_loader:
        by = create_labels(by)
        for i in range(len(bx)):
            if labels_counter[int(by[i])] < data_amount_per_class:
                labels_counter[int(by[i])] += 1
                x0.append(bx[i])
                y0.append(by[i])
            if (labels_counter[0] >= data_amount_per_class) and (labels_counter[1] >= data_amount_per_class):
                got_enough = True
                break
        if got_enough:
            break
    x0, y0 = torch.stack(x0), torch.stack(y0)
    return x0, y0


def load_fashion_data(args):
    data_loader = load_fashion(root=args.datasets_dir, batch_size=100, train=True, shuffle=False, start=0, end=50000)
    x0, y0 = get_balanced_data(args, data_loader, args.data_amount)

    assert not args.data_use_test or (args.data_use_test and args.data_test_amount >= 2)
    data_loader = load_fashion(root=args.datasets_dir, batch_size=100, train=False, shuffle=False, start=0, end=10000)
    x0_test, y0_test = get_balanced_data(args, data_loader, args.data_test_amount)

    x0, y0 = move_to_type_device(x0, y0, args.device)
    x0_test, y0_test = move_to_type_device(x0_test, y0_test, args.device)
    print(f'BALANCE: 0: {y0[y0 == 0].shape[0]}, 1: {y0[y0 == 1].shape[0]}')
    return [(x0, y0)], [(x0_test, y0_test)], None


def get_dataloader(args):
    args.input_dim = 28 * 28
    args.num_classes = 2
    args.output_dim = 1
    args.dataset = 'fashion'

    if args.run_mode == 'reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000

    return load_fashion_data(args)
