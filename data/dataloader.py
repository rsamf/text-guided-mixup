import torch
import json
from random import choices
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler, RandomSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torch.nn.functional as F
import os
from PIL import Image
from data.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from data.ImageNetClasses import IMAGENET_CLASSES
from data.iNaturalist18Classes import get_class_names as get_inat_class_names
from data.PlacesClasses import PLACES_CLASSES
from functools import partial

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

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip()
])

# ImageNet/Places Dataset Class
class LT_Dataset(Dataset):
    def __init__(self, root, txt, dataset, class_names, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.classes = class_names

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        self.targets = self.labels

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
    
    def get_freq(self):
        return torch.tensor(self.img_num_per_cls)

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

    def get_num_classes(self):
        return len(self.classes)
    
    def get_lang_inputs(self):
        text_inputs = [(f"a photo of a {c}") for c in self.classes]
        return text_inputs
    
    def get_class_subdivisions(self):
        def subdivision(num_samples):
            if num_samples > 100:
                return "many"
            elif num_samples > 20:
                return "med"
            else:
                return "few"
        
        return [subdivision(num_samples) for num_samples in self.img_num_per_cls]

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
def pair_local_samples(mixer, batch):
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
    # x = torch.stack([x_i, x_j])
    # y = torch.stack([y_i, y_j])
    # idx = [idx_i, idx_j]
    x, y, _ = mixer.mix(x_i, y_i, x_j, y_j)
    return x, y

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
    elif dataset == 'ImageNet':
        txt = './data/%s/%s_%s.txt'%(dataset, dataset, phase)
        set_ = LT_Dataset(data_root, txt, dataset, IMAGENET_CLASSES, transform=transform)
    elif dataset == 'iNaturalist18':
        txt = './data/%s/%s_%s.txt'%(dataset, dataset, phase)
        set_ = LT_Dataset(data_root, txt, dataset, get_inat_class_names(), transform=transform)
    elif dataset == 'Places':
        txt = './data/%s/%s_%s.txt'%(dataset, dataset, phase)
        set_ = LT_Dataset(data_root, txt, dataset, PLACES_CLASSES, transform=transform)
    else:
        set_ = None
    return set_

def pair_and_mix(batch):
    return pair_local_samples(batch)

def get_dataloader(dataset, batch_size, mixer=None, p_matrix=None, multi_gpu=False, drop_last=False):
    num_workers = os.cpu_count()
    if p_matrix != None:
        pair_and_mix = partial(pair_local_samples, mixer)
        sampler = LocalClassSampler(dataset, p_matrix, multi_gpu)
        return DataLoader(dataset, 
                        sampler=sampler,
                        batch_size=batch_size*2,
                        num_workers=num_workers,
                        collate_fn=pair_and_mix)
    else:
        if multi_gpu:
            num_workers = num_workers // 3
            print(f"Num workers: {num_workers}")
            return DataLoader(dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    sampler=DistributedSampler(dataset),
                    drop_last=drop_last)
        else:
            return DataLoader(dataset,
                    num_workers=num_workers,
                    batch_size=batch_size)
