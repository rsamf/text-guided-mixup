import matplotlib.pyplot as plt
import numpy as np

alpha_ablation_x = [0, .2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.]
alpha_ablation_y = [
    ([50.9, 52.2, 52.6, 54.1, 53.6, 54.7, 53.6, 53.2, 53.0, 53.7, 52.8], "all"),
    ([64.7, 65.3, 62.9, 60.7, 58.0, 55.5, 54.0, 53.2, 52.1, 50.3, 49.8], "many"),
    ([53.1, 53.4, 54.7, 56.9, 59.8, 59.4, 60.7, 60.2, 59.8, 60.9, 61.6], "med"),
    ([33.1, 36.2, 38.4, 43.5, 41.2, 48.4, 45.3, 45.5, 46.7, 49.5, 46.6], "few"),
]

tau_ablation_x = [.002, .01, .05, .25, 1.25, 6.25, 31.25]
tau_ablation_y = [
    ([58.8, 64.8, 80.1, 79.2, 79.6, 79.4, 78.9], "all"),
    ([45.6, 52.3, 83.4, 84.5, 84.9, 84.1, 84.9], "many"),
    ([70.0, 73.9, 83.3, 80.5, 80.4, 80.4, 79.9], "med"),
    ([61.6, 69.0, 72.8, 71.7, 72.6, 73.2, 71.0], "few"),
]

def create_graph(title, x_label, y_label, x, y, other_f=[]):
    plt.clf()
    plt.cla()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for line in y:
        plt.plot(x, line[0], label=line[1], linestyle="--" if line[1] != "all" else "-")
    for f in other_f:
        f(plt)
    plt.xticks(x, x)
    plt.grid()
    plt.legend()
    plt.savefig(f"plots/{x_label}.png")

create_graph("Effect on Different Values for Alpha", "Alpha", "Top-1 Accuracy", alpha_ablation_x, alpha_ablation_y)
create_graph("Effect on Different Values for Tau", "Tau", "Top-1 Accuracy", tau_ablation_x, tau_ablation_y, [lambda plt: plt.xscale('log', base=5)])

### T-SNE
import torch
import random
import os
import clip
import torchvision
from sklearn.manifold import TSNE
torch.set_grad_enabled(False)

encoders, preprocess = clip.load("ViT-B/32", device="cpu")

def get_text_features(text_input):
    with torch.no_grad():
        text_input = torch.cat([clip.tokenize(text) for text in text_input])
        clip_features = encoders.encode_text(text_input).to(dtype=torch.float)
        norm = clip_features.norm(dim=-1, keepdim=True)
        clip_features = clip_features / norm
    return clip_features

dataset = torchvision.datasets.CIFAR100(root="./dataset/CIFAR100", train=False)
examples_labels = ['apple', 'pear', 'lobster', 'crab', 'snake', 'worm', 'bed', 'couch', 'bicycle', 'motorcycle']

def find_indices(input, element):
    indices = [i for i,el in enumerate(input) if el == element]
    return indices

def create_tsne_scatter(x, y, labels, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_x, cls_y, cls_label in zip(x, y, labels):
        ax.scatter(cls_x, cls_y, alpha=.8, label=cls_label, s=90)
    fig.legend()
    fig.savefig(f"plots/{title}.png")

examples_indices = [dataset.classes.index(ex) for ex in examples_labels]
img_max = 50
examples_image_idx = [find_indices(dataset.targets, tgt)[:img_max] for tgt in examples_indices]
min_num_images = 2
cls_num = 10
gamma = 1 / 50
img_num_per_cls = [ int(img_max * (gamma ** (cls_idx / (cls_num - 1.0)))) for cls_idx in range(cls_num) ]
examples_image_idx = [examples_image_idx[i][:img_num] for i,img_num in enumerate(img_num_per_cls)]
examples_images = [torch.stack([preprocess(dataset[img_idx][0]) for img_idx in class_indices]) for class_indices in examples_image_idx]

all_f_v = []
for class_samples in examples_images:
    f_v = encoders.encode_image(class_samples)
    all_f_v.append(f_v)
f_v = torch.cat(all_f_v, dim=0)
print(f_v.shape)

tsne = TSNE(random_state=1, metric="cosine")
embs = tsne.fit_transform(f_v.view(-1, 512))
# Add to dataframe for convenience
x = torch.tensor(embs[:, 0])
y = torch.tensor(embs[:, 1])
print(x.shape, y.shape)
x_view = []
y_view = []
for i in range(len(img_num_per_cls)):
    start = 0 if i == 0 else img_num_per_cls[i-1]
    end = img_num_per_cls[i]
    x_view.append(x[start:end])
    y_view.append(y[start:end])

print(len(x_view), len(y_view), x_view[0].shape)

create_tsne_scatter(x_view, y_view, examples_labels, "Image Feature Space")

## After training
from models.simple import SimpleCLIPModel 
model = SimpleCLIPModel(device="cpu", backbone="ViT-B/16").to("cpu")
model.load_state_dict(torch.load("params/decoupled_single_gpu_1.pt", map_location="cpu"))
model.eval()
f_l = get_text_features(examples_labels)
all_f_v = []
for class_samples in examples_images:
    _, f_v = model(f_l, class_samples, 1)
    all_f_v.append(f_v)
f_v = torch.cat(all_f_v, dim=0)

tsne = TSNE(random_state=1, metric="cosine")
embs = tsne.fit_transform(f_v.view(-1, 512))
# Add to dataframe for convenience
x = torch.tensor(embs[:, 0])
y = torch.tensor(embs[:, 1])
# x = x.view(10, -1)
# y = y.view(10, -1)

create_tsne_scatter(x, y, examples_labels, "Image Feature Space After Training")

