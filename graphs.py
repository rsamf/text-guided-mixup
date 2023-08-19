import matplotlib.pyplot as plt
import json
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

# create_graph("Effect on Different Values for Alpha", "Alpha", "Top-1 Accuracy", alpha_ablation_x, alpha_ablation_y)
# create_graph("Effect on Different Values for Tau", "Tau", "Top-1 Accuracy", tau_ablation_x, tau_ablation_y, [lambda plt: plt.xscale('log', base=5)])

### T-SNE
import torch
import torch.nn.functional as F
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

dataset = torchvision.datasets.CIFAR100(root="./dataset/CIFAR100", train=True)
examples_labels = ['apple', 'pear', 'lobster', 'crab', 'snake', 'worm', 'bed', 'couch', 'bicycle', 'motorcycle']

def find_indices(input, element):
    indices = [i for i,el in enumerate(input) if el == element]
    return indices

def create_tsne_scatter(x, y, z, labels, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7,7))
    lines = []
    lens = [len(cls_x) for cls_x in x]
    for cls_x, cls_y, cls_z, cls_label in zip(x, y, z, labels):
        print(cls_label, len(cls_x))
        lines.append( ax.scatter(cls_x, cls_y, cls_z, alpha=.3) )

    u = np.linspace(0, 2 * np.pi, 130)
    v = np.linspace(0, np.pi, 70)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Use 3x the stride, no scipy zoom
    ax.plot_surface(x, y, z, rstride=3, cstride=3, color='white', edgecolor='#cccccc20', shade=False, alpha=0)

    # Comment out when generating scatter plots
    legend_fig = plt.figure("Legend plot")
    labels = [f"{label} ({len})" for label, len in zip(labels, lens)]
    leg = legend_fig.legend(lines, labels, loc='center')
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    legend_fig.savefig('plots/scatter_legend.png', bbox_inches='tight')

    fig.tight_layout()
    ax.set_aspect('equal')
    ax.axis('off')
    # fig.savefig(f"plots/{title}.png", bbox_inches='tight')
    fig.show()

examples_indices = [dataset.classes.index(ex) for ex in examples_labels]
examples_image_idx = [find_indices(dataset.targets, tgt) for tgt in examples_indices]

with open("cls_freq/CIFAR-100-LT_IMBA100.json", "r") as f:
    freqs = json.load(f)
example_freqs = [freqs[idx] for idx in examples_indices]
img_num_per_cls = example_freqs
print(img_num_per_cls)

cls_num = len(examples_labels)
examples_image_idx = [ examples_image_idx[i][:img_num] for i,img_num in enumerate(img_num_per_cls)]
examples_images = [torch.stack([preprocess(dataset[img_idx][0]) for img_idx in class_indices]) for class_indices in examples_image_idx]

def get_model_tsne(model, title):
    all_f_v = []
    for class_samples in examples_images:
        f_v = model(class_samples)
        all_f_v.append(f_v)
    f_v = torch.stack(all_f_v)

    tsne = TSNE(random_state=1, metric="cosine")
    embs = tsne.fit_transform(f_v.view(-1, 512))
    # Add to dataframe for convenience
    x = torch.tensor(embs[:, 0])
    y = torch.tensor(embs[:, 1])
    x = x.view(10, -1)
    y = y.view(10, -1)

    create_tsne_scatter(x, y, examples_labels, title)

## After training
from models.simple import SimpleCLIPModel 
model = SimpleCLIPModel(device="cpu", backbone="ViT-B/32").to("cpu")
model.eval()
f_l = get_text_features(examples_labels)

def model_fn(img):
    _, f_v = model(f_l, img, 1)
    return f_v

# get_model_tsne(encoders.encode_image, "Image Feature Space")
# model.load_state_dict(torch.load("decoupled_single_gpu_0_0.pt", map_location="cpu"))
# get_model_tsne(model_fn, "Image Feature Space After Training")
# model.load_state_dict(torch.load("decoupled_single_gpu_1_0.pt", map_location="cpu"))
# get_model_tsne(model_fn, "Image Feature Space After Training Phase 1")


## Sample Probability
from data import dataloader
from utils import get_sample_probability_matrix_softmax

data_root = {
    'ImageNet': './dataset/ImageNet',
    'iNaturalist18': './dataset/iNaturalist18',
    "Places": "./dataset/Places",
    'CIFAR10': './dataset/CIFAR10',
    'CIFAR100': './dataset/CIFAR100',
}

tau = {
    'CIFAR10': .05,
    'CIFAR100': .05,
    'ImageNet': .05,
    "Places": .5,
}

def create_sample_plot(title, x_label, y_label, x, y, name):
    plt.clf()
    plt.cla()
    # plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for line in y:
        plt.plot(x, line[0], label=line[1])
    plt.grid()
    plt.legend()
    plt.savefig(f"plots/{name}.png")

def get_sample_prob(dataset_str, imb):
    dataset = dataloader.get_dataset(data_root[dataset_str], dataset_str, 'train', None, cifar_imb_ratio=imb)

    language_input = dataset.get_lang_inputs()
    p_matrix = get_sample_probability_matrix_softmax(model.get_text_features, language_input, tau[dataset_str], dataset.classes)
    freq = dataset.get_freq()
    p_i = freq / len(dataset)
    p_i, idx_p_i = torch.sort(p_i, descending=True)
    # [100, 100] * [1, 100]
    p_matrix = p_matrix[:, idx_p_i]
    mul = p_matrix * p_i
    p_j = torch.sum( mul, dim=1 )
    p = (p_i + p_j) / 2
    y = (p_i, "LT"), (p, "Local Sampling")
    imb = p_i.max()/p_i.min()
    new_imb = p.max() / p.min()
    print(f"New imbalance factor: {new_imb}")
    create_sample_plot(f"{dataset_str} Sample Probability with imb factor {imb}", "y", "p(y)", range(freq.shape[0]), y, f"{dataset_str}imb={imb}")

get_sample_prob("CIFAR100", 100)
get_sample_prob("CIFAR100", 50)
get_sample_prob("CIFAR100", 10)
get_sample_prob("CIFAR10", 100)
get_sample_prob("CIFAR10", 50)
get_sample_prob("CIFAR10", 10)
get_sample_prob("ImageNet", None)
get_sample_prob("Places", None)
