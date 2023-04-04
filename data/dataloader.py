import torch
import json
import numpy as np
from random import choices
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
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

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key == 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


# Dataset
class LT_Dataset(Dataset):
    def __init__(self, root, txt, dataset, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        # save the class frequency
        if 'train' in txt:
            if not os.path.exists('cls_freq'):
                os.makedirs('cls_freq')
            freq_path = os.path.join('cls_freq', dataset + '.json')
            self.img_num_per_cls = [0 for _ in range(max(self.labels)+1)]
            for cls in self.labels:
                self.img_num_per_cls[cls] += 1
            with open(freq_path, 'w') as fd:
                json.dump(self.img_num_per_cls, fd)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index

class LocalClassLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers, probability_matrix=None):
        super(LocalClassLoader, self).__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=self.collate_fn)
        self.probability_matrix = probability_matrix
        self.num_classes = self.dataset.get_num_classes()
        
        if self.probability_matrix != None:
            self.samplers = []
            targets = torch.tensor(self.dataset.targets)
            for i in range(self.num_classes):
                possible_alt_img = (targets == i).nonzero()
                self.samplers.append(SubsetRandomSampler(possible_alt_img).__iter__())

    def collate_fn(self, batch):
        imgs1 = []
        tgts1 = []
        indexes1 = []
        imgs2 = []
        tgts2 = []
        indexes2 = []
        for x1, y1, i1 in batch:
            imgs1.append(x1)
            tgts1.append(y1)
            indexes1.append(i1)
            if self.probability_matrix != None:
                prob_dist = self.probability_matrix[y1]
                y2 = choices(range(self.num_classes), prob_dist)[0]
                i2 = next(self.samplers[y2])
                x2 = self.dataset[i2][0]
                imgs2.append(x2)
                tgts2.append(y2)
                indexes2.append(i2)
        imgs1 = torch.stack(imgs1)
        tgts1 = torch.from_numpy(np.asarray(tgts1))

        if self.probability_matrix != None:
            imgs2 = torch.stack(imgs2)
            tgts2 = torch.from_numpy(np.asarray(tgts2))
            return imgs1, tgts1, indexes1, imgs2, tgts2, indexes2
        return imgs1, tgts1, indexes1

def get_dataset(data_root, dataset, phase, cifar_imb_ratio=None, transform=None):
    if phase == 'train_plain':
        txt_split = 'train'
    elif phase == 'train_val':
        txt_split = 'val'
        phase = 'train'
    else:
        txt_split = phase
    txt = './data/%s/%s_%s.txt'%(dataset, dataset, txt_split)
    print('Loading data from %s' % (txt))

    if dataset == 'iNaturalist18':
        print('===> Loading iNaturalist18 statistics')
        key = 'iNaturalist18'
    else:
        key = 'default'

    if dataset == 'CIFAR10':
        print('====> CIFAR10 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR10(phase, imbalance_ratio=cifar_imb_ratio, root=data_root, transform=transform)
    elif dataset == 'CIFAR100':
        print('====> CIFAR100 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR100(phase, imbalance_ratio=cifar_imb_ratio, root=data_root, transform=transform)
    else:
        rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']
        if phase not in ['train', 'val']:
            transform = get_data_transform('test', rgb_mean, rgb_std, key)
        else:
            transform = get_data_transform(phase, rgb_mean, rgb_std, key)

        print('Use data transformation:', transform)
        set_ = LT_Dataset(data_root, txt, dataset, transform)
    return set_

def get_dataloader(dataset, batch_size, num_workers=4, p_matrix=None):
    if p_matrix != None:
        return LocalClassLoader(dataset, batch_size=batch_size, num_workers=num_workers, probability_matrix=p_matrix)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
