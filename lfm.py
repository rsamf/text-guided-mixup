import torch
from torch.distributions.beta import Beta

class LocalFeatureMixup():
    def __init__(self, alpha, freq, prob_sample, cls_sampler):
        self.alpha = alpha
        self.beta_dist = Beta(torch.tensor([.5]), torch.tensor([.5]))
        self.freq = freq
        self.prob_sample = prob_sample

    def mix(self, x, y, n):
        x_i, x_j = x
        y_i, y_j = y
        n_i, n_j = n
        lambda_x = self.beta_dist.sample()
        lambda_y = lambda_x + self.alpha * (n_i - n_j) / (n_i + n_j)
        x_gen = lambda_x * x_i + (1 - lambda_x) * x_j
        y_gen = lambda_y * y_i + (1 - lambda_y) * y_j
        return x_gen, y_gen
    
    # def setup_sample(self, i, x_i, y_i):
    #     j = i #TODO: sample j from prob dist
    #     x_j, y_j = self.cls_sampler(j)
    #     x = x_i, x_j
    #     y = y_i, y_j
    #     n = self.freq[i], self.freq[j]
    #     return x, y, n
    
    def get_x_y(self, x, y, n):
        # x, y, n = self.setup_sample(i, x_i, y_i)
        x_lfm, y_lfm = self.mix(x, y, n)
        return x_lfm, y_lfm
