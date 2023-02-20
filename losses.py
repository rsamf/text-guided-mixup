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
    def __init__(self, tgt_map=None, freq_path="cls_freq/CIFAR-100-LT_IMBA100.json"):
        super(BalancedCE, self).__init__()
        with open(freq_path, 'r') as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        if tgt_map != None:
            new_freq = torch.zeros(20).to(DEVICE)
            for i, sc_index in enumerate(tgt_map):
                new_freq[sc_index] += freq[i]
            freq = new_freq
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

# class WCFC(_Loss):
#     def __init__(self, feature_mapping='type1'):
#         super(WCFC, self).__init__()
#         self.feature_mapping = feature_mapping
#         self.cos = nn.CosineSimilarity(dim=0)

#     def forward(self, features, weights):
#         if self.feature_mapping == 'type1':
#             # normalization of averaged features
#             mean_features = torch.mean(features, dim=1)
#             # g = mean_features / torch.norm(mean_features)
#         # else:
#         #     norm_features = torch.norm(features)
#         #     sum_norm_features = torch.sum(norm_features)
#         #     g = sum_norm_features / torch.norm(sum_norm_features)
#         print(features.shape)
#         print(weights.shape)
#         print(mean_features.shape)
#         # print(g.shape)
#         return -torch.log(self.cos(mean_features, weights))
#         # return -torch.log( torch.dot(g, (weights / torch.norm(weights))) )

# class AWS(_Loss):
#     def __init__(self, margin=.2):
#         super(AWS, self).__init__()
#         self.margin = margin
#         self.cos = nn.CosineSimilarity(dim=0)

#     def ang_dist(self, w_i, w_j):
#         return self.cos(w_i, w_j)

#     def indicator(self, ang_dist):
#         # if weights are close enough (within margin), then enable
#         return (ang_dist > self.margin).to(dtype=torch.long)

#     def forward(self, W):
#         N = W.shape[0]
#         sum = 0
#         sum_still_in_margin = 0
#         for i in range(N):
#             for j in range(N):
#                 ang = self.ang_dist(W[i], W[j]) if i != j else 0
#                 ind = self.indicator(ang)
#                 sum += -torch.log(-ang * ind + 1)
#                 sum_still_in_margin += ind
#         return sum / sum_still_in_margin

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

# class GeometricLoss(_Loss):
#     def __init__(self, a=1,b=1,c=1):
#         super(GeometricLoss, self).__init__()
#         self.a = a
#         self.b = b
#         self.c = c
#         self.cls_loss = BalancedSoftmax()
#         self.wcfc_loss = WCFC()
#         self.aws_loss = AWS()

#     def forward(self, input, target, features, weights):
#         return self.a*self.cls_loss(input, target) + self.b*self.wcfc_loss(features, weights) + self.c*self.aws_loss(weights)
