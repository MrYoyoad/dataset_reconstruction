
def setup_problem(args):
    if False:
        pass
    elif args.problem == 'cifar10_vehicles_animals':
        from problems.cifar10_vehicles_animals import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'mnist_odd_even':
        from problems.mnist_odd_even import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'fashion_odd_even':
        from problems.fashion_odd_even import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'mnist_10class':
        from problems.mnist_10class import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'fashion_10class':
        from problems.fashion_10class import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'flowers102_parity':
        from problems.flowers102_parity import get_dataloader
        return get_dataloader(args)
    else:
        raise ValueError(f'Unknown args.problem={args.problem}')
    return data_loader

