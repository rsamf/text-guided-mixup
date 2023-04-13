import argparse
from utils import DEVICE, Validator, get_sample_probability_matrix, get_text_distances
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

def get_freq():
    freq_path="cls_freq/CIFAR-100-LT_IMBA100.json" # TODO
    with open(freq_path, 'r') as fd:
        freq = json.load(fd)
        freq = torch.tensor(freq)
    return freq

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--lr', type=int, default=None)
parser.add_argument('--use_lfm', default=False, action='store_true')
parser.add_argument('--model_dir', type=str, default=None)

args = parser.parse_args()
yml = yaml.load(Path(args.cfg).read_text(), yaml.Loader) if args.cfg != None else {}

epochs = args.epochs or yml.get("epochs")
batch_size = args.batch_size or yml.get("batch_size")
loss_str = args.loss or yml.get("loss")
dataset_str = args.dataset or yml.get("dataset")
lr = args.lr or yml.get("lr")
use_lfm = args.use_lfm or yml.get("use_lfm")
alpha = yml.get("alpha")

freq = get_freq()

def setup_loss_fn(loss_str, model, language_input):
    if loss_str == 'CE':
        return CrossEntropyLoss(reduction='mean')
    if loss_str == 'BalCE':
        return losses.BalancedCE(freq, reduction='mean')
    if loss_str == 'BalBCE':
        return losses.BCE_loss(reduction='mean')
    if loss_str == 'Focal':
        return losses.FocalLoss()
    if loss_str == 'LDAM':
        return losses.LDAMLoss(freq, reduction='mean')
    if loss_str == 'MMS':
        return losses.MarginMetricSoftmax(get_text_distances(model.language_model, language_input), reduction='mean')

def main():
    dr = data_root[dataset_str]
    model = SimpleCLIPModel().to(DEVICE)
    train_set = dataloader.get_dataset(dr, dataset_str, 'train', cifar_imb_ratio=100, transform=model.preprocess)
    val_set = dataloader.get_dataset(dr, dataset_str, 'val', transform=model.preprocess)
    # Get Language Input and Sample Probability Matrix
    p_matrix = None
    if use_lfm:
        language_input = train_set.get_lang_inputs()
        p_matrix = get_sample_probability_matrix(model.language_model, language_input)

    train_loader = dataloader.get_dataloader(train_set, batch_size, num_workers=4, p_matrix=p_matrix)
    val_loader = dataloader.get_dataloader(val_set, batch_size, num_workers=4)
    loss_fn = setup_loss_fn(loss_str, model, train_set.get_lang_inputs())
    validator = Validator(model, val_set, val_loader, train_set.get_class_subdivisions(), loss_fn)
    date = datetime.now().strftime('%b%d-%H-%M-%S')
    writer = SummaryWriter(f'runs/{loss_str}-{date}')
    trainer.train(model, train_set, train_loader, validator, loss_fn, epochs, lr, use_lfm, alpha, freq, writer)

main()
