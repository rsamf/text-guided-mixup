import argparse
<<<<<<< HEAD
from utils import DEVICE, Validator, get_sample_probability_matrix_softmax, get_text_distances
from train import decoupled_trainer_mgpu
=======
from utils import get_sample_probability_matrix_softmax, get_text_distances
from train import decoupled_trainer, decoupled_trainer_mgpu
>>>>>>> b0a0a02c5287bcc8cf7cac3f0249b97c4ef449e3
from models.simple import SimpleCLIPModel
from data import dataloader
from torch.nn import CrossEntropyLoss
import os
import torch
import losses
import yaml
import json
from pathlib import Path
from datetime import datetime
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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
parser.add_argument('--cfg', nargs='+', default=None, type=str, required=True)
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--lr', type=int, default=None)
parser.add_argument('--use_lfm', default=False, action='store_true')
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--multi_gpu', type=bool, default=False, action='store_true')

args = parser.parse_args()
yml = {}
for cfg in args.cfg:
    print(f"using config {cfg}")
    cfg = yaml.load(Path(cfg).read_text(), yaml.Loader)
    yml.update(cfg)
print(yml)

epochs = args.epochs or yml.get("epochs")
batch_size = args.batch_size or yml.get("batch_size")
loss_str = args.loss or yml.get("loss")
dataset_str = args.dataset or yml.get("dataset")
lr = args.lr or yml.get("lr")
use_lfm = args.use_lfm or yml.get("use_lfm")
multi_gpu = args.multi_gpu or yml.get("multi_gpu")
alpha = yml.get("alpha")
backbone = yml.get("backbone")
phase1_model = yml.get("phase1_model")

def setup_loss_fn(loss_str, model, language_input, freq):
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
        return losses.MarginMetricSoftmax(get_text_distances(model.get_text_features, language_input), reduction='mean')

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def multi_gpu_train(device, num_gpus, model, train_set, train_loader, val_loader, loss_fn, epochs, lr, alpha, freq, logdir, phase1_model):
    ddp_setup(device, num_gpus)
    writer = SummaryWriter(logdir)
    model = model.to(device)
    model = DDP(model, device_ids=[device])
    decoupled_trainer_mgpu.train(model, device, train_set, train_loader, val_loader, loss_fn, epochs, lr, alpha, freq, writer, phase1_model)
    destroy_process_group()

def main():
    dr = data_root[dataset_str]
    model = SimpleCLIPModel(backbone)
    train_set = dataloader.get_dataset(dr, dataset_str, 'train', model.preprocess, cifar_imb_ratio=100)
    val_set = dataloader.get_dataset(dr, dataset_str, 'val', model.preprocess)
    # Get Language Input and Sample Probability Matrix
    p_matrix = None
    if use_lfm:
        language_input = train_set.get_lang_inputs()
        p_matrix = get_sample_probability_matrix_softmax(model.get_text_features, language_input, train_set.classes)

    freq = get_freq()
    loss_fn = setup_loss_fn(loss_str, model, train_set.get_lang_inputs(), freq)
    date = datetime.now().strftime('%b%d-%H-%M-%S')
    logdir = f'runs/{loss_str}-{date}'
    if multi_gpu:
        num_gpus = torch.cuda.device_count()
        train_loader_lfm = dataloader.get_dataloader(train_set, batch_size, p_matrix=p_matrix, multi_gpu=True)
        train_loader = [train_loader_lfm, train_loader_lfm]
        val_loader = dataloader.get_dataloader(val_set, batch_size, multi_gpu=True)
        mp.spawn(multi_gpu_train, args=(num_gpus, model, train_set, train_loader, val_loader, loss_fn, epochs, lr, alpha, freq, logdir, phase1_model), nprocs=num_gpus)

    else:
        writer = SummaryWriter(logdir)
        decoupled_trainer.train(model, train_set, train_loader, val_loader, loss_fn, epochs, lr, alpha, freq, writer, phase1_model)

main()
