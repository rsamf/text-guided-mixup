import argparse
import torch
from utils import DEVICE
from train import decoupled_sc
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
    model = SimpleCLIPModel(100).to(DEVICE)
    dr = data_root['CIFAR100']
    decoupled_sc.train(model, dr)
    val_set, val_loader = dataloader.load_data(dr, 'CIFAR100_LT', 'val', 4, num_workers=4, transform=model.preprocess)
    validate(model, val_set, val_loader)

def validate(model, val_set, val_loader):
    with torch.no_grad():
        language_features = model.language_model(val_set.get_lang_inputs())

    all_preds = []
    for x, tgt, _ in val_loader:
        x = x.to(DEVICE)
        tgt = tgt.to(device="cuda")
        similarity = model(language_features, x)
        preds = torch.argmax(similarity, dim=-1) == tgt
        all_preds.append(preds)
    all_preds = torch.stack(all_preds)
    acc = torch.sum(all_preds) / tgt.shape[0]
    print(acc)

main()
