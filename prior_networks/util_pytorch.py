import os
import re
from pathlib import Path
from typing import Union
# import context.py
import numpy as np
import torch

from prior_networks import models
from prior_networks.datasets import image

# TODO Add LeNet for MNIST and MNIST-like stuff
MODEL_DICT = {'vgg11': models.vgg11,
              'vgg11_bn': models.vgg11_bn,
              'vgg13': models.vgg13,
              'vgg13_bn': models.vgg13_bn,
              'vgg16': models.vgg16,
              'vgg16_bn': models.vgg16_bn,
              'vgg19': models.vgg19,
              'vgg19_bn': models.vgg19_bn,
              'resnet18': models.resnet18,
              'resnet34': models.resnet34,
              'resnet50': models.resnet50,
              'resnet101': models.resnet101,
              'resnet152': models.resnet152,
              'resnext50_32x4d': models.resnext50_32x4d,
              'resnext101_32x8d': models.resnext101_32x8d,
              'wide_resnet50_2': models.wide_resnet50_2,
              'wide_resnet101_2': models.wide_resnet101_2,
              'wide_resnet28_10': models.wide_resnet28_10,
              'densenet121': models.densenet121,
              'densenet161': models.densenet161,
              'densenet169': models.densenet169,
              'densenet201': models.densenet201}

DATASET_DICT = {'MNIST': image.MNIST,
                'KMNIST': image.KMNIST,
                'FMNIST': image.FashionMNIST,
                'EMNIST': image.EMNIST,
                'SVHN': image.SVHN,
                'CIFAR10': image.CIFAR10,
                'CIFAR100': image.CIFAR100,
                'ImageNet': image.ImageNet}


def save_model(model: torch.nn.Module, n_in: int, n_channels: int,
               num_classes: int, arch: str, small_inputs: bool, path: Union[Path, str]):
    assert arch in MODEL_DICT.keys()
    torch.save({'num_classes': num_classes,
                'n_in': n_in,
                'n_channels': n_channels,
                'model_state_dict': model.state_dict(),
                'arch': arch,
                'small_inputs': small_inputs}, os.path.join(path, 'model.tar'))


def categorical_entropy(probs, axis=1, keepdims=False):
    """

    :param probs:
    :param axis:
    :param keepdims:
    :return:
    """
    return -np.sum(probs * np.log(probs, out=np.zeros_like(probs), where=(probs != 0.)), axis=axis,
                   keepdims=keepdims)


def categorical_entropy_torch(probs, dim=1, keepdim=False):
    """Calculate categorical entropy purely in torch"""
    log_probs = torch.log(probs)
    log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
    entropy = -torch.sum(probs * log_probs, dim=dim, keepdim=keepdim)
    return entropy


def get_grid(xrange=(-500, 500), yrange=(-500, 500), resolution=200, dtype=np.float32):
    x = np.linspace(*xrange, resolution, dtype=dtype)
    y = np.linspace(*yrange, resolution, dtype=dtype)
    xx, yy = np.meshgrid(x, y, sparse=False)
    return xx, yy


def get_grid_eval_points(xrange, yrange, res):
    xx, yy = get_grid(xrange, yrange, res, dtype=np.float32)
    eval_points = torch.from_numpy(np.stack((xx.ravel(), yy.ravel()), axis=1))
    return eval_points


def select_device(device_name):
    if device_name is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device_name = device_name.strip()
        if re.search("^cuda:[0-9]$", device_name):
            assert torch.cuda.is_available()
            # Assert that the device selected isn't out of range:
            assert torch.cuda.device_count() > int(device_name[-1])
            device = torch.device(device_name)
            print(f"Using cuda device: {torch.cuda.get_device_name(device)} | {device_name}")
        elif device_name != "cpu":
            raise AttributeError(f"No such device allowed: {device_name}")
        device = torch.device(device_name)
    return device


def select_gpu(gpu_id: int):
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > gpu_id
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using device: {torch.cuda.get_device_name(device)} unit {gpu_id}.")
    else:
        print(f"Using CPU device.")
        device = torch.device("cpu")

    return device


#
# # def KL_divergence(probs1, probs2, epsilon=1e-10):
# #     return np.sum(probs1*(np.log(probs1+epsilon)-np.log(probs2+epsilon)), axis=1)
#
# # def expected_pairwise_KL_divergence(probs):
# #     KL = 0.0
# #     for i in range(probs.shape[2]):
# #         for j in range(probs.shape[2]):
# #             KL += KL_divergence(probs[:,:,i], probs[:,:,j])
# #     return KL
#
#
# def test_accuracy(predict_func, dataset, batch_size=100):
#     n_correct = 0  # Track the number of correct classifications
#     testloader = DataLoader(dataset, batch_size=batch_size,
#                             shuffle=False, num_workers=1)
#
#     with torch.no_grad():
#         for i, data in enumerate(testloader, 0):
#             inputs, labels = data
#             probs = predict_func(inputs)
#             n_correct += torch.sum(torch.argmax(probs, dim=1) == labels).item()
#     accuracy = n_correct / len(testloader.dataset)
#
#     return accuracy
#
#
# def test_error_rate(predict_func, dataset, batch_size=100):
#     return 1. - test_accuracy(predict_func, dataset, batch_size=batch_size)
#
#
# def test_nll(log_predict_func, dataset, batch_size=100):
#     loss_fun = torch.nn.NLLLoss(reduction='sum')
#     tot_loss = 0
#     testloader = DataLoader(dataset, batch_size=batch_size,
#                             shuffle=False, num_workers=1)
#     with torch.no_grad():
#         for i, data in enumerate(testloader, 0):
#             inputs, labels = data
#             probs = log_predict_func(inputs)
#             tot_loss += loss_fun(probs, labels).item()
#
#     mean_nll = tot_loss / len(testloader.dataset)
#     return mean_nll
#
##
#
# def cartestian_to_barometric(coord):
#     """Transform a set of cartesian coordinates to barometric. Assumes last dimension represents (x, y)
#     coordinates"""
#     corners = (np.array([0, 0]), np.array([1, 0]), np.array([0.5, 0.75 ** 0.5]))
#     barom = np.stack((np.linalg.norm(coord - corners[0], axis=1),
#                       np.linalg.norm(coord - corners[1], axis=1),
#                       np.linalg.norm(coord - corners[2], axis=1)), axis=1)
#     return barom
#
#
# def ensemble_mutual_information(probs):
#     """Calculate mutual information of ensemble predictions"""
#     mean_probs = np.mean(probs, axis=2)
#
#     entropy_mean = categorical_entropy(mean_probs)
#
#     entropies = categorical_entropy(probs)
#     mean_entropies = np.mean(entropies, axis=1)
#
#     mutual_info = entropy_mean - mean_entropies
#     return mutual_info


class TargetTransform:
    def __init__(self, target_concentration, gamma, ood=False):
        self.target_concentration = target_concentration
        self.gamma = gamma
        self.ood = ood

    def __call__(self, label):
        return self.forward(label)

    def forward(self, label):
        if self.ood:
            return (0, self.target_concentration, self.gamma)
        else:
            return (label, self.target_concentration, self.gamma)
