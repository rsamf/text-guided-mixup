import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from utils import DEVICE

class BalancedCE(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, freq_path="cls_freq/CIFAR-100-LT_IMBA100.json"):
        super(BalancedCE, self).__init__()
        with open(freq_path, 'r') as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        self.sample_per_class = freq

    def forward(self, input, labels, reduction='mean'):
        """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
        Args:
        labels: A int tensor of size [batch].
        logits: A float tensor of size [batch, no_of_classes].
        sample_per_class: A int tensor of size [no of classes].
        reduction: string. One of "none", "mean", "sum"
        Returns:
        loss: A float tensor. Balanced Softmax Loss.
        """
        spc = self.sample_per_class.type_as(input)
        spc = spc.unsqueeze(0).expand(input.shape[0], -1)
        logits = input + spc.log()
        loss = F.cross_entropy(logits, target=labels, reduction=reduction)
        return loss

class AFS(_Loss):
    def __init__(self, margin=.6):
        super(AFS, self).__init__()
        self.margin = torch.tensor(margin).to(DEVICE)
        self.cos = nn.CosineSimilarity(dim=0)

    def ang_dist(self, w_i, w_j):
        return self.cos(w_i, w_j)

    def indicator(self, ang_dist):
        # if weights are close enough (within margin), then enable
        return (ang_dist > self.margin).to(dtype=torch.long)

    def forward(self, W):
        N = W.shape[0]
        sum = 0
        sum_still_in_margin = 0
        for i in range(N):
            for j in range(N):
                ang = self.ang_dist(W[i], W[j]) if i != j else 0
                ind = self.indicator(ang)
                sum += -torch.log(-ang * ind + 1)
                sum_still_in_margin += ind
        return sum / sum_still_in_margin if sum_still_in_margin > 0 else 0
