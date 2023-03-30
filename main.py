import argparse
from utils import DEVICE, Validator
from train import trainer
from models.simple import SimpleCLIPModel
from data import dataloader
from torch.nn import CrossEntropyLoss
import losses
import yaml
from pathlib import Path

data_root = {
    'ImageNet': './dataset/ImageNet',
    'Places': './dataset/Places-LT',
    'iNaturalist18': '/checkpoint/bykang/iNaturalist18',
    'CIFAR10': './dataset/CIFAR10',
    'CIFAR100': './dataset/CIFAR100',
}

loss_choices = {
    'CE': CrossEntropyLoss,
    'BalCE': losses.BalancedCE,
    'BalBCE': losses.BCE_loss,
    'Focal': losses.FocalLoss,
    'LDAM': losses.LDAMLoss
}

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
# parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--lr', type=int, default=None)
# parser.add_argument('--test_open', default=False, action='store_true')
# parser.add_argument('--output_logits', default=False)
parser.add_argument('--model_dir', type=str, default=None)
# parser.add_argument('--save_feat', type=str, default='')

args = parser.parse_args()
yml = yaml.load(Path(args.cfg).read_text(), yaml.Loader) if args.cfg != None else {}

epochs = args.epochs or yml.get("epochs")
batch_size = args.batch_size or yml.get("batch_size")
loss_str = args.loss or yml.get("loss")
dataset_str = args.dataset or yml.get("dataset")
lr = args.lr or yml.get("lr")

def main():
    model = SimpleCLIPModel().to(DEVICE)
    dr = data_root[dataset_str]
    loss_fn = loss_choices[loss_str]()
    train_set, train_loader = dataloader.load_data(dr, dataset_str, 'train', batch_size, num_workers=4, shuffle=True, cifar_imb_ratio=100, transform=model.preprocess)
    val_set, val_loader = dataloader.load_data(dr, dataset_str, 'val', batch_size, num_workers=4, transform=model.preprocess)
    validator = Validator(model, val_set, val_loader, train_set.get_class_subdivisions())
    trainer.train(model, train_set, train_loader, validator, loss_fn, epochs, lr)

main()
