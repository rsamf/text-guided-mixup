import argparse
import torch
from utils import DEVICE, Evaluator, Validator
# from train import decoupled_sc
from train import cross_entropy
from models.simple import SimpleCLIPModel
from data import dataloader

data_root = {'ImageNet': './dataset/ImageNet',
             'Places': './dataset/Places-LT',
             'iNaturalist18': '/checkpoint/bykang/iNaturalist18',
             'CIFAR10': './dataset/CIFAR10',
             'CIFAR100': './dataset/CIFAR100',
             }

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--save_feat', type=str, default='')

args = parser.parse_args()


def main():
    model = SimpleCLIPModel().to(DEVICE)
    dr = data_root['CIFAR100']
    # decoupled_sc.train(model, dr)
    train_set, train_loader = dataloader.load_data(dr, 'CIFAR100_LT', 'train', 4, num_workers=4, shuffle=True, cifar_imb_ratio=100, transform=model.preprocess)
    val_set, val_loader = dataloader.load_data(dr, 'CIFAR100_LT', 'val', 4, num_workers=4, transform=model.preprocess)
    validator = Validator(model, val_set, val_loader, train_set.get_class_subdivisions())
    cross_entropy.train(model, train_set, train_loader, validator)




main()
