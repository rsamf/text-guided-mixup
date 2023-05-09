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
    
# https://github.com/XuZhengzhuo/LiVT/blob/68546ef189c486caa271066d8bfa25ec214192df/util/loss.py
class BCE_loss(nn.Module):

    def __init__(self, args,
                target_threshold=None, 
                type=None,
                reduction='mean', 
                pos_weight=None):
        super(BCE_loss, self).__init__()
        self.lam = 1.
        self.K = 1.
        self.smoothing = args.smoothing
        self.target_threshold = target_threshold
        self.weight = None
        self.pi = None
        self.reduction = reduction
        self.register_buffer('pos_weight', pos_weight)

        if type == 'Bal':
            self._cal_bal_pi(args)
        if type == 'CB':
            self._cal_cb_weight(args)

    def _cal_bal_pi(self, args):
        cls_num = torch.Tensor(args.cls_num)
        self.pi = cls_num / torch.sum(cls_num)

    def _cal_cb_weight(self, args):
        eff_beta = 0.9999
        effective_num = 1.0 - np.power(eff_beta, args.cls_num)
        per_cls_weights = (1.0 - eff_beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(args.cls_num)
        self.weight = torch.FloatTensor(per_cls_weights).to(args.device)

    def _bal_sigmod_bias(self, x):
        pi = self.pi.to(x.device)
        bias = torch.log(pi) - torch.log(1-pi)
        x = x + self.K * bias
        return x

    def _neg_reg(self, labels, logits, weight=None):
        if weight == None:
            weight = torch.ones_like(labels).to(logits.device)
        pi = self.pi.to(logits.device)
        bias = torch.log(pi) - torch.log(1-pi)
        logits = logits * (1 - labels) * self.lam + logits * labels # neg + pos
        logits = logits + self.K * bias
        weight = weight / self.lam * (1 - labels) + weight * labels # neg + pos
        return logits, weight

    def _one_hot(self, x, target):
        num_classes = x.shape[-1]
        off_value = self.smoothing / num_classes
        on_value = 1. - self.smoothing + off_value
        target = target.long().view(-1, 1)
        target = torch.full((target.size()[0], num_classes),
            off_value, device=x.device, 
            dtype=x.dtype).scatter_(1, target, on_value)
        return target

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            target = self._one_hot(x, target)
        if self.target_threshold is not None:
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        weight = self.weight
        if self.pi != None: x = self._bal_sigmod_bias(x)
        # if self.lam != None:
        #     x, weight = self._neg_reg(target, x)
        C = x.shape[-1] # + log C
        return C * F.binary_cross_entropy_with_logits(
                    x, target, weight, self.pos_weight,
                    reduction=self.reduction)

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
        ce = F.cross_entropy((pred + offset[labels]) / self.temp, labels)
        return ce
