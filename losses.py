import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np

class BalancedCE(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, freq, reduction='mean'):
        super(BalancedCE, self).__init__()
        self.sample_per_class = freq
        self.reduction = reduction

    def forward(self, input, labels):
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
        loss = F.cross_entropy(logits, target=labels, reduction=self.reduction)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        cross_entropy = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        p = torch.exp(-cross_entropy)
        loss = (1 - p) ** self.gamma * cross_entropy
        return loss.mean()

# https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
class LDAMLoss(nn.Module):
    def __init__(self, freq, max_m=0.5, weight=None, s=30, reduction='mean'):
        super(LDAMLoss, self).__init__()
        freq = np.array(freq)
        m_list = 1.0 / np.sqrt(np.sqrt(freq))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.from_numpy(m_list).type(torch.float)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.reduction=reduction

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        self.m_list = self.m_list.to(device=x.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight, reduction=self.reduction)

class MarginMetricSoftmax(_Loss):
    def __init__(self, text_distances, l=.3, temp=.01):
        super(MarginMetricSoftmax, self).__init__()
        self.logits_offset = l*text_distances
        self.temp = temp

    def forward(self, pred, labels):
        offset = self.logits_offset.type(pred.type())
        offset = offset.to(device=pred.device)
        if labels.dim() == 1:
            ce = F.cross_entropy((pred + offset[labels]) / self.temp, labels)
        elif labels.dim() == 2:
            max_v, max_i = torch.topk(labels, 2, dim=1)
            scale0, label0 = max_v[:, 0], max_i[:, 0]
            scale1, label1 = max_v[:, 1], max_i[:, 1]
            ce0 = F.cross_entropy((pred + offset[label0]) / self.temp, label0, reduction='none')
            ce1 = F.cross_entropy((pred + offset[label1]) / self.temp, label1, reduction='none')
            ce = (scale0 * ce0 + scale1 * ce1).mean()
        else:
            raise ValueError(f"Label dimensions expected to be <= 2 but received {labels.dim()}")
        return ce
