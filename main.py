import argparse
from utils import get_sample_probability_matrix_softmax, get_text_distances, get_text_similarities
from train import decoupled_trainer, decoupled_trainer_mgpu
from models.simple import SimpleCLIPModel
from data import dataloader
from torch.nn import CrossEntropyLoss
import os
import torch
import losses
import yaml
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from mixups import LocalFeatureMixup

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

data_root = {
    'ImageNet': './dataset/ImageNet',
    'iNaturalist18': './dataset/iNaturalist18',
    "Places": "./dataset/Places",
    'CIFAR10': './dataset/CIFAR10',
    'CIFAR100': './dataset/CIFAR100',
}

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', nargs='+', default=None, type=str, required=True)
parser.add_argument('--run_exp', default=None, type=str, required=False)
parser.add_argument('--gpu', type=int, required=False)
parser.add_argument('--checkpoint', default=None, type=str, required=False)
args = parser.parse_args()

class ExperimentRunner():
    def __init__(self, config_file, arg_dict, proc):
        self.arg_dict = arg_dict
        self.proc = proc
        self.cfg = yaml.load(Path(config_file).read_text(), yaml.Loader)
        self.key = self.cfg["key"]
        if self.cfg.get("values") != None:
            self.values = self.cfg.get("values")
        else:
            start = self.cfg["start"]
            end = self.cfg["end"]
            step = self.cfg["step"]
            self.values = np.arange(start, end, step)

    def run(self):
        print(f"Using experiment runner on {len(self.values)} experiments")
        for value in self.values:
            self.arg_dict[self.key] = value
            print(self.arg_dict)
            self.proc(self.arg_dict)

def setup_loss_fn(loss_str, model, language_input, freq):
    if loss_str == 'CE':
        return CrossEntropyLoss(reduction='mean')
    if loss_str == 'BalCE':
        return losses.BalancedCE(freq, reduction='mean')
    if loss_str == 'Focal':
        return losses.FocalLoss()
    if loss_str == 'LDAM':
        return losses.LDAMLoss(freq, reduction='mean')
    if loss_str == 'MMS':
        return losses.MarginMetricSoftmax(get_text_distances(model.get_text_features, language_input))

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    print(f"Setting up GPU {rank}")
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def multi_gpu_train(device, num_gpus, backbone, train_set, val_set, batch_size, p_matrix, f_l, loss_fn, epochs, lr, alpha, freq, logdir):
    ddp_setup(device, num_gpus)
    writer = SummaryWriter(logdir)
    model = SimpleCLIPModel(device, backbone).to(device)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    mixer = LocalFeatureMixup(alpha, freq) if alpha != None else None

    train_loader = dataloader.get_dataloader(train_set, batch_size, p_matrix=p_matrix, mixer=mixer, multi_gpu=True)
    train_loaders = [train_loader, train_loader]
    val_loader = dataloader.get_dataloader(val_set, batch_size*num_gpus, multi_gpu=True, drop_last=True)
    f_l = f_l.to(device)
    decoupled_trainer_mgpu.train(model, device, num_gpus, train_set, train_loaders, val_loader, f_l, loss_fn, epochs, lr, mixer, writer, args.checkpoint, main_device=1)
    destroy_process_group()

def single_gpu_train(device, backbone, train_set, val_set, batch_size, p_matrix, f_l, loss_fn, epochs, lr, alpha, freq, logdir):
    writer = SummaryWriter(logdir)
    model = SimpleCLIPModel(device, backbone).to(device)
    mixer = LocalFeatureMixup(alpha, freq) if alpha != None else None

    train_loader = dataloader.get_dataloader(train_set, batch_size, mixer=mixer, p_matrix=p_matrix)
    train_loaders = [train_loader, train_loader]
    val_loader = dataloader.get_dataloader(val_set, batch_size)
    f_l = f_l.to(device)
    decoupled_trainer.train(model, device, train_set, train_loaders, val_loader, f_l, loss_fn, epochs, lr, mixer, writer, args.checkpoint)

def main(yml):
    epochs = yml.get("epochs")
    batch_size = yml.get("batch_size")
    loss_str = yml.get("loss")
    dataset_str = yml.get("dataset")
    lr = yml.get("lr")
    use_lfm = yml.get("use_lfm")
    alpha = yml.get("alpha")
    tau = yml.get("tau")
    cifar_imb = yml.get("cifar_imb")
    backbone = yml.get("backbone")

    dr = data_root[dataset_str]
    setup_model = SimpleCLIPModel("cpu", backbone)
    train_set = dataloader.get_dataset(dr, dataset_str, 'train', setup_model.preprocess, cifar_imb_ratio=cifar_imb)
    val_set = dataloader.get_dataset(dr, dataset_str, 'val', setup_model.preprocess)
    # Get Language Input and Sample Probability Matrix
    p_matrix = None
    if use_lfm:
        language_input = train_set.get_lang_inputs()
        p_matrix = get_sample_probability_matrix_softmax(setup_model.get_text_features, language_input, tau, train_set.classes)
    else:
        alpha = None

    with torch.no_grad():
        f_l = setup_model.get_text_features(train_set.get_lang_inputs())
        f_l_norm = f_l.norm(dim=-1, keepdim=True)
        f_l = f_l / f_l_norm

    freq = train_set.get_freq()
    print(f"imbalance ratio is set to be:", freq.max()/freq.min())
    loss_fn = setup_loss_fn(loss_str, setup_model, train_set.get_lang_inputs(), freq)
    date = datetime.now().strftime('%b%d-%H-%M-%S')
    logdir = f'runs/{loss_str}-{date}'
    num_gpus = torch.cuda.device_count() - 1 # Dont use last one
    if num_gpus > 1 and args.gpu == None:
        print("Multi GPU Training")
        mp.spawn(multi_gpu_train, args=(num_gpus, backbone, train_set, val_set, batch_size, p_matrix, f_l, loss_fn, epochs, lr, alpha, freq, logdir), nprocs=num_gpus)
    else:
        print("Single GPU Training")
        if args.gpu != None:
            gpu = args.gpu
        else:
            gpu = 0
        single_gpu_train(gpu, backbone, train_set, val_set, batch_size, p_matrix, f_l, loss_fn, epochs, lr, alpha, freq, logdir)

if __name__ == "__main__":
    yml = {}
    for cfg in args.cfg:
        print(f"Using config {cfg}")
        cfg = yaml.load(Path(cfg).read_text(), yaml.Loader)
        yml.update(cfg)
    print(yml)

    if args.run_exp:
        runner = ExperimentRunner(args.run_exp, yml, main)
        runner.run()
    else:
        main(yml)

