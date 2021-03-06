import context
import argparse
import os
import sys
import pathlib
from pathlib import Path
import math
import numpy as np

import torch
from torch.utils import data

from prior_networks.datasets.toy.classification.mog import MixtureOfGaussiansDataset
from prior_networks.priornet.dpn_losses import DirichletKLLoss, PriorNetMixedLoss
from prior_networks.util_pytorch import DATASET_DICT, select_gpu
from prior_networks.priornet.training import TrainerWithOOD
from prior_networks.util_pytorch import TargetTransform, choose_optimizer
from torch import optim
from prior_networks.datasets.image.standardised_datasets import construct_transforms
from prior_networks.models.model_factory import ModelFactory
from prior_networks.priornet.run.synth_model import SynthModel


parser = argparse.ArgumentParser(description='Train a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision '
                                             'dataset.')
parser.add_argument('--n_epochs', type=int,
                    help='How many epochs to train for.')
parser.add_argument('--lr', type=float,
                    help='Initial learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.95, help='LR decay multiplies')
parser.add_argument('--lrc', action='append', type=int, help='LR decay milestones')
parser.add_argument('--model_dir', type=str, default='./',
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('--target_concentration', type=float, default=1e2,
                    help='Target in-domain concentration.')
parser.add_argument('--concentration', type=float, default=1.0,
                    help='Concentration of non-target classes.')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='Weight for OOD loss.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='L2 weight decay.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training.')
parser.add_argument('--model_load_path', type=str, default='./model',
                    help='Source where to load the model from.')
parser.add_argument('--reverse_KL', type=bool, default=True,
                    help='Whether to use forward or reverse KL. Default is to ALWAYS use reverse KL.')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specify which GPUs to to run on.')
parser.add_argument('--optimizer', choices=['SGD', 'ADAM'], default='ADAM',
                    help='Choose which optimizer to use.')
parser.add_argument('--augment',
                    action='store_true',
                    help='Whether to use augmentation.')
parser.add_argument('--rotate',
                    action='store_true',
                    help='Whether to use rotation augmentation')
parser.add_argument('--jitter', type=float, default=0.0,
                    help='Specify how much random color, '
                         'hue, saturation and contrast jitter to apply')
parser.add_argument('--normalize',
                    action='store_false',
                    help='Whether to standardize input (x-mu)/std')
parser.add_argument('--resume',
                    action='store_true',
                    help='Whether to resume training from checkpoint.')
parser.add_argument('--clip_norm', type=float, default=10.0,
                    help='Gradient clipping norm value.')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to where to checkpoint.')
parser.add_argument("--std", type=float, choices=[1, 4])


def main():
    args = parser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_train_dpn.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    model_dir = Path(args.model_dir)
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = model_dir / 'model'
    # Check that we are training on a sensible GPU
    print(args.gpu, torch.cuda.device_count())
    assert max(args.gpu) <= torch.cuda.device_count() - 1

    device = select_gpu(args.gpu)
    model = SynthModel()

    if len(args.gpu) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        print('Using Multi-GPU training.')
    model.to(device)

    train_dataset = MixtureOfGaussiansDataset(size=30000, noise=args.std, scale=4)
    val_dataset = MixtureOfGaussiansDataset(size=500, noise=args.std, scale=4)

    ood_dataset = MixtureOfGaussiansDataset(size=30000, noise=args.std, scale=4., OOD=True)
    ood_val_dataset = MixtureOfGaussiansDataset(size=500, noise=args.std, scale=4., OOD=True)

    assert len(val_dataset) == len(ood_val_dataset)
    assert len(train_dataset) == len(ood_dataset)

    # Even out dataset lengths.
    id_ratio = 1.0

    # Set up training and test criteria
    id_criterion = DirichletKLLoss(target_concentration=args.target_concentration,
                                   concentration=args.concentration,
                                   reverse=args.reverse_KL)

    ood_criterion = DirichletKLLoss(target_concentration=0.0,
                                    concentration=args.concentration,
                                    reverse=args.reverse_KL)

    criterion = PriorNetMixedLoss([id_criterion, ood_criterion], mixing_params=[1.0, args.gamma])

    # Select optimizer and optimizer params
    optimizer, optimizer_params = choose_optimizer(args.optimizer,
                                                   args.lr,
                                                   args.weight_decay)

    # Setup model trainer and train model
    lrc = [int(lrc / id_ratio) for lrc in args.lrc]
    trainer = TrainerWithOOD(model=model,
                             criterion=criterion,
                             id_criterion=id_criterion,
                             ood_criterion=ood_criterion,
                             test_criterion=criterion,
                             ood_dataset=ood_dataset,
                             test_ood_dataset=ood_val_dataset,
                             train_dataset=train_dataset,
                             test_dataset=val_dataset,
                             optimizer=optimizer,
                             device=device,
                             checkpoint_path=checkpoint_path,
                             scheduler=optim.lr_scheduler.MultiStepLR,
                             optimizer_params=optimizer_params,
                             scheduler_params={'milestones': lrc, 'gamma': args.lr_decay},
                             batch_size=args.batch_size,
                             clip_norm=args.clip_norm)
    if args.resume:
        try:
            trainer.load_checkpoint(True, True, map_location=device)
        except:
            print('No checkpoint found, training from empty model.')
            pass
    trainer.train(int(args.n_epochs / id_ratio), resume=args.resume)


if __name__ == "__main__":
    main()
