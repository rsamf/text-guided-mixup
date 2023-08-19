import torch
from torch.distributions.beta import Beta
import torch.nn.functional as F
import numpy as np

class Mixup():
    def __init__(self, alpha=1.0):
        self.alpha = 1.0

    def mix(self, x, y):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class Remix():
    def __init__(self, freq, kappa=3, tau=.5):
        self.kappa = kappa
        self.tau = tau
        self.freq = freq
        self.num_classes = self.freq.shape[0]
        self.beta_dist = Beta(torch.tensor([.5]), torch.tensor([.5]))

    def mix(self, x, y):
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        y_i, y_j = y, y[index]
        lambda_x = self.beta_dist.sample()[0]
        mixed_x = lambda_x * x + (1 - lambda_x) * x[index, :]
        n_i, n_j = self.freq[y_i], self.freq[y_j]
        zero_cond = torch.logical_and((n_i / n_j >= self.kappa), (lambda_x < self.tau))
        one_cond = torch.logical_and((n_i / n_j <= 1 / self.kappa), (1 - lambda_x < self.tau))
        lambda_y = torch.where(zero_cond, torch.ones_like(y), torch.where(one_cond, torch.zeros_like(y), lambda_x.repeat(y.shape[0])))
        y_i_onehot = F.one_hot(y_i, num_classes=self.num_classes)
        y_j_onehot = F.one_hot(y_j, num_classes=self.num_classes)
        lambda_y = lambda_y.unsqueeze(1).expand_as(y_i_onehot)
        mixed_y = lambda_y * y_i_onehot + (1 - lambda_y) * y_j_onehot
        return mixed_x, mixed_y

class LocalFeatureMixup():
    def __init__(self, alpha, freq):
        self.alphas = alpha
        self.beta_dist = Beta(torch.tensor([.2]), torch.tensor([.2]))
        self.freq = freq
        self.num_classes = self.freq.shape[0]
        self.set_phase(0)

    def set_phase(self, phase):
        if isinstance(self.alphas, list):
            self.alpha = self.alphas[phase]
        else:
            self.alpha = self.alphas

    def mix(self, x_i, y_i, x_j, y_j):
        y_i_onehot = F.one_hot(y_i, num_classes=self.num_classes)
        y_j_onehot = F.one_hot(y_j, num_classes=self.num_classes)
        n_i, n_j = self.freq[y_i], self.freq[y_j]

        lambda_x = self.beta_dist.sample()[0]
        x_gen = lambda_x * x_i + (1 - lambda_x) * x_j
        # Generate y target
        y_offset = self.alpha * (n_i - n_j) / (n_i + n_j)
        lambda_y = torch.clamp(lambda_x - y_offset, 0, 1)
        lambda_y = lambda_y.unsqueeze(1).expand_as(y_i_onehot)
        y_gen = lambda_y * y_i_onehot + (1 - lambda_y) * y_j_onehot
        y_no_offset = lambda_x * y_i_onehot + (1 - lambda_x) * y_j_onehot
        return x_gen, y_gen, y_no_offset
