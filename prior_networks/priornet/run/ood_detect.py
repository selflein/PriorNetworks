#! /usr/bin/env python

import context
import argparse
import os
import sys
import numpy as np
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F

from prior_networks.assessment.ood_detection import eval_ood_detect
from prior_networks.evaluation import eval_logits_on_dataset
from prior_networks.datasets.image import construct_transforms
from prior_networks.priornet.dpn import dirichlet_prior_network_uncertainty
from prior_networks.util_pytorch import DATASET_DICT, select_gpu
from prior_networks.models.model_factory import ModelFactory

from uncertainty_eval.metrics import brier_decomposition, brier_score, classification_calibration

matplotlib.use('agg')

parser = argparse.ArgumentParser(description='Evaluates model predictions and uncertainty '
                                             'on in-domain test data')

parser.add_argument('data_path', type=str,
                    help='absolute path to training data.')
parser.add_argument('id_dataset', choices=DATASET_DICT.keys(),
                    help='Specify name of the in-dimain dataset to evaluate model on.')
parser.add_argument('--ood_dataset',
                    help='Specify name of the out-of-domain dataset to evaluate model on.', action="append")
parser.add_argument('output_path', type=str,
                    help='Path of directory for saving model outputs.')
parser.add_argument('--model_dir', type=str, default='./',
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for processing')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specify which GPUs to to run on.')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite a previous run of this script')


def main():
    args = parser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/ood_detect.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')
    if os.path.isdir(args.output_path) and not args.overwrite:
        print(f'Directory {args.output_path} exists. Exiting...')
        sys.exit()
    elif os.path.isdir(args.output_path) and args.overwrite:
        os.remove(args.output_path + '/*')
    else:
        os.makedirs(args.output_path)

    # Check that we are using a sensible GPU device
    device = select_gpu(args.gpu)

    model_dir = Path(args.model_dir)
    # Load up the model
    ckpt = torch.load(model_dir, map_location=device)
    model = ModelFactory.model_from_checkpoint(ckpt)

    model.to(device)
    model.eval()

    # Load the in-domain evaluation data
    id_dataset = DATASET_DICT[args.id_dataset](root=args.data_path,
                                               transform=construct_transforms(n_in=ckpt['n_in'],
                                                                              mean=DATASET_DICT[args.id_dataset].mean,
                                                                              std=DATASET_DICT[args.id_dataset].std,
                                                                              mode='eval'),
                                               target_transform=None,
                                               download=True,
                                               split='test')

    # Evaluate the model
    id_logits, id_labels = eval_logits_on_dataset(model=model,
                                                dataset=id_dataset,
                                                batch_size=args.batch_size,
                                                device=device)

    id_labels, id_probs, id_logits = id_labels.numpy(), F.softmax(id_logits,
                                                                dim=1).numpy(), id_logits.numpy()

    accuracy = np.mean(np.squeeze(id_labels) == np.argmax(id_probs, axis=-1))
    ece, mce = classification_calibration(id_labels, id_probs)
    brier = brier_score(id_labels, id_probs)
    uncertainty, resolution, reliability = brier_decomposition(id_labels, id_probs)
    with open(os.path.join(args.output_path, 'results.txt'), 'a') as f:
        f.write(f"Accuracy: {accuracy * 100:.02f}\n")
        f.write(f"ECE: {ece * 100:.02f}\n")
        f.write(f"MCE: {mce * 100:.02f}\n")
        f.write(f"Brier: {brier * 100:.02f}\n")
        f.write(f"Brier UNC: {uncertainty * 100:.02f}\n")
        f.write(f"Brier RES: {resolution * 100:.02f}\n")
        f.write(f"Brier REL: {reliability * 100:.02f}\n")

    id_uncertainties = dirichlet_prior_network_uncertainty(id_logits)

    for ood_dataset in args.ood_dataset:
        save_dir = os.path.join(args.output_path, ood_dataset)
        os.mkdir(save_dir)

        with open(os.path.join(args.output_path, 'results.txt'), 'a') as f:
            f.write(f"\n\n{ood_dataset}\n")

        split = ood_dataset.split("_", maxsplit=1)
        if len(split) == 2 and split[1] == "unscaled":
            unscaled_ds = True
            ood_dataset_name = split[0]
        else:
            unscaled_ds = False
            ood_dataset_name = ood_dataset

        ood_dataset = DATASET_DICT[ood_dataset_name](root=args.data_path,
                                                    transform=construct_transforms(n_in=ckpt['n_in'],
                                                                                    mean=DATASET_DICT[args.id_dataset].mean,
                                                                                    std=DATASET_DICT[args.id_dataset].std,
                                                                                    mode='eval',
                                                                                    unscaled=unscaled_ds),
                                                    target_transform=None,
                                                    download=True,
                                                    split='test')
        print(f"ID dataset length: {len(id_dataset)}, OOD dataset length: {len(ood_dataset)}")


        ood_logits, ood_labels = eval_logits_on_dataset(model=model,
                                                        dataset=ood_dataset,
                                                        batch_size=args.batch_size,
                                                        device=device)

        ood_labels, ood_probs, ood_logits = ood_labels.numpy(), F.softmax(ood_logits,
                                                                        dim=1).numpy(), ood_logits.numpy()

        ood_uncertainties = dirichlet_prior_network_uncertainty(ood_logits)

        # Compute Labels
        in_domain = np.zeros_like(id_labels)
        out_domain = np.ones((ood_labels.shape[0],))
        domain_labels = np.concatenate((in_domain, out_domain), axis=0)

        eval_ood_detect(domain_labels=domain_labels,
                        in_uncertainties=id_uncertainties,
                        out_uncertainties=ood_uncertainties,
                        save_path=save_dir)

if __name__ == '__main__':
    main()
