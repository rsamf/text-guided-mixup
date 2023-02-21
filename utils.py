import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
