"""
Adopted from https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch
"""
import os
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import json
from PIL import Image

aquatic_mammals = ['beaver', 'dolphin', 'otter', 'seal', 'whale']
fish =            ['aquarium_fish', 'flatfish', 'ray', 'shark','trout']
flowers =         ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']
food_containers = ['bottle', 'bowl', 'can', 'cup', 'plate']
fruits_and_vegetables = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
household_electrical_devices = ['clock', 'keyboard', 'lamp', 'telephone', 'television']
household_furniture = ['bed', 'chair', 'couch', 'table', 'wardrobe']
insects = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']
large_carnivores = ['bear', 'leopard', 'lion', 'tiger', 'wolf']
large_man_made_outdoor_things = ['bridge', 'castle', 'house', 'road', 'skyscraper']
large_natural_outdoor_scenes = ['cloud', 'forest', 'mountain', 'plain', 'sea']
large_omnivores_and_herbivores = ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']
medium_sized_mammals = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']
non_insect_invertebrates = ['crab', 'lobster', 'snail', 'spider', 'worm']
people = ['baby', 'boy', 'girl', 'man', 'woman']
reptiles = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']
small_mammals = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
trees = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']
vehicles_1 = ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train']
vehicles_2 = ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']

super_classes = [
    aquatic_mammals, fish, flowers, food_containers, fruits_and_vegetables,
    household_electrical_devices, household_furniture, insects, large_carnivores, large_man_made_outdoor_things,
    large_natural_outdoor_scenes, large_omnivores_and_herbivores, medium_sized_mammals, non_insect_invertebrates, people,
    reptiles, small_mammals, trees, vehicles_1, vehicles_2
]

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    dataset_name = 'CIFAR-10-LT'

    def __init__(self, phase, imbalance_ratio, root, imb_type='exp', transform=None):
        train = True if phase == "train" else False
        super(IMBALANCECIFAR10, self).__init__(root, train, transform=None, target_transform=None, download=True)
        self.train = train
        if self.train:
            self.img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(self.img_num_per_cls)

        if transform:
            self.transform = transform
        
        self.labels = self.targets

        print("{} Mode: Contain {} images".format(phase, len(self.data)))

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        gamma = 1. / imb_factor
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (gamma ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * gamma))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        # save the class frequency
        if not os.path.exists('cls_freq'):
            os.makedirs('cls_freq')
        freq_path = os.path.join('cls_freq', self.dataset_name + '_IMBA{}.json'.format(imb_factor))
        with open(freq_path, 'w') as fd:
            json.dump(img_num_per_cls, fd)
        self.img_num_per_cls = img_num_per_cls

        return img_num_per_cls

    def get_freq(self):
        return torch.tensor(self.img_num_per_cls)

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
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


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    cls_num = 100
    s_cls_num = 20
    dataset_name = 'CIFAR-100-LT'
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def get_super_class_mapping(self):
        # Row is class, column is superclass
        index_mapping = [[torch.tensor(self.classes.index(c)) for c in super_class] for super_class in super_classes]
        cls_per_super_class = self.cls_num / self.s_cls_num
        linear_mapping = torch.stack([ 
            torch.sum(torch.stack([F.one_hot(c, num_classes=self.cls_num) for c in super_class]), dim=0) 
            for super_class in index_mapping 
        ]) / cls_per_super_class

        def find_super_class_i(cls):
            for i in range(len(super_classes)):
                if cls in super_classes[i]:
                    return torch.tensor(i)
            raise Exception(f"Couldn't find class {cls} in list of super classes")

        tgt_mapping = [ find_super_class_i(c) for c in self.classes ]
        return linear_mapping.T, tgt_mapping
