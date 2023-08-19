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

def get_tnse_output(model):
    all_f_v = []
    for class_samples in examples_images:
        f_v = model(class_samples)
        all_f_v.append(f_v)
    f_v = torch.cat(all_f_v, dim=0)

    tsne = TSNE(n_components=3, random_state=1, metric="cosine", n_iter=2000)
    embs = tsne.fit_transform(f_v.view(-1, 512))
    embs = torch.tensor(embs)
    embs = F.normalize(embs, dim=-1)
    x = embs[:, 0]
    y = embs[:, 1]
    z = embs[:, 2]
    x_view = []
    y_view = []
    z_view = []
    ptr = 0
    for i in range(cls_num):
        img_num = img_num_per_cls[i]
        x_view.append(x[ptr:ptr+img_num])
        y_view.append(y[ptr:ptr+img_num])
        z_view.append(z[ptr:ptr+img_num])
        ptr += img_num
    return x_view, y_view, z_view

# x, y, z = get_tnse_output(encoders.encode_image)
# torch.save((x,y,z), "tempo0.pt")
x, y, z = torch.load("tempo0.pt")
create_tsne_scatter(x, y, z, examples_labels, "feat")

## After training
from models.simple import SimpleCLIPModel 
model = SimpleCLIPModel(device="cpu", backbone="ViT-B/16").to("cpu")
model.eval()


## After training with CE
model.load_state_dict(torch.load("params/cifar_100_ce/decoupled_single_gpu_0_0.pt", map_location="cpu"))
x, y, z = get_tnse_output(model.get_visual_features)
torch.save((x,y,z), "tempo-ce.pt")
create_tsne_scatter(x, y, z, examples_labels, "feat-ce")


## After training with LFM + CE
model.load_state_dict(torch.load("params/decoupled_single_gpu_1.pt", map_location="cpu"))
# x, y, z = get_tnse_output(model.get_visual_features)
# torch.save((x,y,z), "tempo1.pt")
x, y, z = torch.load("tempo1.pt")
create_tsne_scatter(x, y, z, examples_labels, "feat-lfm-ce")

