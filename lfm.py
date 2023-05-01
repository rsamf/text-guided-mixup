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

        lambda_x = self.beta_dist.sample()[0]
        x_gen = lambda_x * x_i + (1 - lambda_x) * x_j
        # Generate y target
        y_offset = self.alpha * (n_i - n_j) / (n_i + n_j)
        lambda_y = torch.clamp(lambda_x - y_offset, 0, 1)
        lambda_y = lambda_y.unsqueeze(1).expand_as(y_i_onehot)
        y_gen = lambda_y * y_i_onehot + (1 - lambda_y) * y_j_onehot
        y_no_offset = lambda_x * y_i_onehot + (1 - lambda_x) * y_j_onehot
        # print(y_gen)
        return x_gen, y_gen, y_no_offset
    
    def mix_features(self, f_i, y_i, f_j, y_j):
        y_i_onehot = F.one_hot(y_i, num_classes=self.num_classes)
        y_j_onehot = F.one_hot(y_j, num_classes=self.num_classes)
        n_i, n_j = self.freq[y_i], self.freq[y_j]

        lambda_x = self.beta_dist.sample()[0]
        f_gen = lambda_x * f_i + (1 - lambda_x) * f_j
        f_gen_norm = f_gen.norm(dim=-1, keepdim=True)
        f_gen = f_gen / f_gen_norm
        # Generate y target
        y_offset = self.alpha * (n_i - n_j) / (n_i + n_j)
        lambda_y = torch.clamp(lambda_x - y_offset, 0, 1)
        lambda_y = lambda_y.unsqueeze(1).expand_as(y_i_onehot)
        y_gen = lambda_y * y_i_onehot + (1 - lambda_y) * y_j_onehot
        return f_gen, y_gen
