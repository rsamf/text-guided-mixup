import torch
import json
import numpy as np
from random import choices
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler, RandomSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torch.nn.functional as F
import os
from PIL import Image
from data.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100

# Image statistics
RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std':[0.229, 0.224, 0.225]
    }
}

# # Data transformation with augmentation
# def get_data_transform(split, rgb_mean, rbg_std, key='default'):
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(rgb_mean, rbg_std)
#         ]) if key == 'iNaturalist18' else transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
#             transforms.ToTensor(),
#             transforms.Normalize(rgb_mean, rbg_std)
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(rgb_mean, rbg_std)
#         ]),
#         'test': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(rgb_mean, rbg_std)
#         ])
#     }
#     return data_transforms[split]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0)
])

# # Dataset
# class LT_Dataset(Dataset):
#     def __init__(self, root, txt, dataset):
#         self.img_path = []
#         self.labels = []

#         with open(txt) as f:
#             for line in f:
#                 self.img_path.append(os.path.join(root, line.split()[0]))
#                 self.labels.append(int(line.split()[1]))

#         # save the class frequency
#         if 'train' in txt:
#             if not os.path.exists('cls_freq'):
#                 os.makedirs('cls_freq')
#             freq_path = os.path.join('cls_freq', dataset + '.json')
#             self.img_num_per_cls = [0 for _ in range(max(self.labels)+1)]
#             for cls in self.labels:
#                 self.img_num_per_cls[cls] += 1
#             with open(freq_path, 'w') as fd:
#                 json.dump(self.img_num_per_cls, fd)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
#         path = self.img_path[index]
#         label = self.labels[index]

#         with open(path, 'rb') as f:
#             sample = Image.open(f).convert('RGB')

#         if self.transform is not None:
#             sample = self.transform(sample)

#         return sample, label, index

class ResettableSubsetSampler(SubsetRandomSampler):
    def __init__(self, indices):
        super(ResettableSubsetSampler, self).__init__(indices)
        self.reset()
        self.sample_count = 0

    def __iter__(self):
        for sample in self.iterator:
            yield sample

    def sample(self):
        if self.sample_count == len(self):
            self.reset()
        self.sample_count += 1
        return next(self.iterator)

    def reset(self):
        self.iterator = super().__iter__()
        self.sample_count = 0

class LocalClassSampler(Sampler):
    def __init__(self, dataset, probability_matrix, multi_gpu):
        super(LocalClassSampler, self).__init__(dataset)
        self.random_sampler = DistributedSampler(dataset) if multi_gpu else RandomSampler(dataset)
        self.dataset = dataset
        self.probability_matrix = probability_matrix
        self.num_classes = self.dataset.get_num_classes()
        self.samplers = []
        self.targets = torch.tensor(self.dataset.targets)
        for i in range(self.num_classes):
            possible_alt_img = (self.targets == i).nonzero()
            self.samplers.append(ResettableSubsetSampler(possible_alt_img))

    def get_next_j_sample(self, i):
        prob_dist = self.probability_matrix[i]
        j = choices(range(self.num_classes), prob_dist)[0]
        j_index = self.samplers[j].sample()
        return j_index
    
    def set_epoch(self, epoch):
        self.random_sampler.set_epoch(epoch)

    def __iter__(self):
        random_sampler = self.random_sampler.__iter__()
        for i_index in random_sampler:
            i = self.dataset.targets[i_index]
            j_index = self.get_next_j_sample(i)
            yield i_index
            yield j_index # collate_fn will need to capture the odd indices

    def __len__(self):
        return self.num_samples * 2

# Output: [2, B, 3, 224, 224]
def pair_local_samples(batch):
    x_i = []
    x_j = []
    y_i = []
    y_j = []
    idx_i = []
    idx_j = []
    i = 0
    for x, y, idx in batch:
        if i % 2 == 0:
            x_i.append(x)
            y_i.append(y)
            idx_i.append(idx)
        else:
            x_j.append(x)
            y_j.append(y)
            idx_j.append(idx)
        i+=1
    x_i = torch.stack(x_i)
    x_j = torch.stack(x_j)
    y_i = torch.tensor(y_i)
    y_j = torch.tensor(y_j)
    x = torch.stack([x_i, x_j])
    y = torch.stack([y_i, y_j])
    idx = [idx_i, idx_j]
    return x, y, idx

def get_dataset(data_root, dataset, phase, model_preprocess, cifar_imb_ratio=None):
    transform = None
    if phase == "train":
        transform = transforms.Compose([
            TRAIN_TRANSFORMS,
            model_preprocess
        ])
    else:
        transform = model_preprocess

    if dataset == 'CIFAR10':
        print('====> CIFAR10 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR10(phase, imbalance_ratio=cifar_imb_ratio, root=data_root, transform=transform)
    elif dataset == 'CIFAR100':
        print('====> CIFAR100 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR100(phase, imbalance_ratio=cifar_imb_ratio, root=data_root, transform=transform)
    else:
        set_ = None
    return set_

def get_dataloader(dataset, batch_size, p_matrix=None, multi_gpu=False):
    if p_matrix != None:
        sampler = LocalClassSampler(dataset, p_matrix, multi_gpu)
        return DataLoader(dataset, 
                        sampler=sampler,
                        batch_size=batch_size*2,
                        collate_fn=pair_local_samples)
    else:
        if multi_gpu:
            return DataLoader(dataset,
                    batch_size=batch_size,
                    sampler=DistributedSampler(dataset))
        else:
            return DataLoader(dataset, 
                    batch_size=batch_size)
