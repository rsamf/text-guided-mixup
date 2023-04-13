import torch
from torch.distributions.beta import Beta
import torch.nn.functional as F

class LocalFeatureMixup():
    def __init__(self, alpha, freq):
        self.alpha = alpha
        self.beta_dist = Beta(torch.tensor([.5]), torch.tensor([.5]))
        self.freq = freq
        self.num_classes = self.freq.shape[0]

    def mix(self, x_i, y_i, x_j, y_j):
        y_i_onehot = F.one_hot(y_i, num_classes=self.num_classes)
        y_j_onehot = F.one_hot(y_j, num_classes=self.num_classes)
        n_i, n_j = self.freq[y_i], self.freq[y_j]
        # print(y_i_onehot.shape, n_i.shape, x_i.shape)
        # Generate x sample
        # lambda_x = self.beta_dist.sample_n(y_i_onehot.shape[0])
        lambda_x = self.beta_dist.sample()[0]
        # print(lambda_x.shape)
        x_gen = lambda_x * x_i + (1 - lambda_x) * x_j
        # Generate y target
        # lambda_x = lambda_x.view(-1,1).expand_as(y_i_onehot)
        y_offset = self.alpha * (n_i - n_j) / (n_i + n_j)
        # 
        lambda_y = torch.clamp(lambda_x + y_offset, 0, 1)
        # lambda_y = lambda_x + y_offset
        lambda_y = lambda_y.unsqueeze(1).expand_as(y_i_onehot)
        # print(lambda_y.shape, y_i_onehot.shape)
        y_gen = lambda_y * y_i_onehot + (1 - lambda_y) * y_j_onehot
        return x_gen, y_gen
