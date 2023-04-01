import argparse
from utils import DEVICE, Validator
from train import trainer
from models.simple import SimpleCLIPModel
from data import dataloader
from torch.nn import CrossEntropyLoss
import torch
import losses
import yaml
import json
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


data_root = {
    'ImageNet': './dataset/ImageNet',
    'Places': './dataset/Places-LT',
    'iNaturalist18': '/checkpoint/bykang/iNaturalist18',
    'CIFAR10': './dataset/CIFAR10',
    'CIFAR100': './dataset/CIFAR100',
}

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--lr', type=int, default=None)
parser.add_argument('--model_dir', type=str, default=None)

args = parser.parse_args()
yml = yaml.load(Path(args.cfg).read_text(), yaml.Loader) if args.cfg != None else {}

epochs = args.epochs or yml.get("epochs")
batch_size = args.batch_size or yml.get("batch_size")
loss_str = args.loss or yml.get("loss")
dataset_str = args.dataset or yml.get("dataset")
lr = args.lr or yml.get("lr")

def get_freq():
    freq_path="cls_freq/CIFAR-100-LT_IMBA100.json" # TODO
    with open(freq_path, 'r') as fd:
        freq = json.load(fd)
        freq = torch.tensor(freq)
    return freq

def setup_loss_fn(loss_str):
    if loss_str == 'CE':
        return CrossEntropyLoss(reduction='mean')
    if loss_str == 'BalCE':
        return losses.BalancedCE(get_freq(), reduction='mean')
    if loss_str == 'BalBCE':
        return losses.BCE_loss(reduction='mean')
    if loss_str == 'Focal':
        return losses.FocalLoss()
    if loss_str == 'LDAM':
        return losses.LDAMLoss(get_freq(), reduction='mean')

def main():
    model = SimpleCLIPModel().to(DEVICE)
    dr = data_root[dataset_str]
    train_set, train_loader = dataloader.load_data(dr, dataset_str, 'train', batch_size, num_workers=4, shuffle=True, cifar_imb_ratio=100, transform=model.preprocess)
    val_set, val_loader = dataloader.load_data(dr, dataset_str, 'val', batch_size, num_workers=4, transform=model.preprocess)
    loss_fn = setup_loss_fn(loss_str)
    validator = Validator(model, val_set, val_loader, train_set.get_class_subdivisions(), loss_fn)
    date = datetime.now().strftime('%b%d-%H-%M-%S')
    writer = SummaryWriter(f'runs/{loss_str}-{date}')
    trainer.train(model, train_set, train_loader, validator, loss_fn, epochs, lr, writer)

main()
