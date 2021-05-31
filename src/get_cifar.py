from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def get_cifar10():

    trainset = CIFAR10(
        root="data",
        train=True,
        download=True
    )
    train_loader = DataLoader(
        trainset, batch_size=100, shuffle=True, num_workers=2
    )

    testset = CIFAR10(
        root="data",
        train=False,
        download=True
    )
    test_loader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    return train_loader, test_loader
