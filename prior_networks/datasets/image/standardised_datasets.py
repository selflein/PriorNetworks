"""
A number of classes that transform the torch vision datasets into a standard format
usable for uncertainty (e.g. out-of-domain vs. in-domain) experimentation.
"""
import context
import torchvision
from torchvision import transforms
from PIL import Image
import torchvision.datasets as datasets

#

split_options = ['train', 'val', 'test']


def construct_transforms(n_in: int, mode: str, augment: bool = False):
    """

    :param n_in:
    :param mode:
    :param augment:
    :return:
    """
    assert mode in ['train', 'eval', 'ood']

    transf_list = []

    if augment:
        if mode == 'eval':
            transf_list.extend([transforms.Resize(n_in, Image.BICUBIC)])
        elif mode == 'train':
            transf_list.extend([transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15, resample=Image.BICUBIC),
                                transforms.Resize(n_in, Image.BICUBIC),
                                transforms.Pad(4, padding_mode='symmetric'),
                                transforms.RandomCrop(n_in)])
        else:
            transf_list.extend([transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(15, resample=Image.BICUBIC),
                                transforms.Resize(n_in, Image.BICUBIC),
                                transforms.Pad(4, padding_mode='symmetric'),
                                transforms.RandomCrop(n_in)])
    else:
        transf_list.extend([transforms.Resize(n_in, Image.BICUBIC)])

    transf_list.extend([transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010))])
    # transf_list.extend([transforms.ToTensor(),
    #                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    # (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261) ???
    # (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
    # (0.499, 0.484, 0.444), (0.258, 0.250, 0.269]) C10+C100
    return transforms.Compose(transf_list)


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         train=train)


class FashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         train=train)


class KMNIST(torchvision.datasets.KMNIST):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         train=train)


class EMNIST(torchvision.datasets.EMNIST):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         split='byclass',
                         target_transform=target_transform,
                         train=train)


class SEMEION(torchvision.datasets.SEMEION):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        print('SEMEION does not have standard test/train splits. '
              'Generating full dataset...')

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform)


class Omniglot(torchvision.datasets.Omniglot):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         background=train)


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         train=train)


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         train=train)


class SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        if split == 'val':
            split = 'test'

        super().__init__(root=root,
                         split=split,
                         download=download,
                         transform=transform,
                         target_transform=target_transform)


class ImageNet(torchvision.datasets.ImageNet):
    def __init__(self, root, transform, target_transform, download, split):
        # TODO Add standard transforms for imagenet
        assert split in split_options
        if split == 'test':
            split = 'val'

        super().__init__(root=root,
                         split=split,
                         download=download,
                         transform=transform,
                         target_transform=target_transform)
