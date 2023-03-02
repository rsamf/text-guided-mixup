import torch
import torch.nn as nn
import torch.linalg as L
import torch.nn.functional as F
from data import dataloader
from models.simple import SimpleCLIPModel
from utils import DEVICE, Evaluator

def get_ang_dist():
    model = SimpleCLIPModel().to(DEVICE)
    dr = './dataset/CIFAR100'
    set, loader = dataloader.load_data(dr, 'CIFAR100_LT', 'val', 4, num_workers=4, shuffle=True, cifar_imb_ratio=100, transform=model.preprocess)
    language_input = set.get_lang_inputs()
    language_model = model.language_model
    S, T = set.get_super_class_mapping()
    S = S.to(device=DEVICE, dtype=torch.float)

    with torch.no_grad():
        f = language_model(language_input)

    f_norm = L.vector_norm(f, dim=1, keepdim=True)
    f = f / f_norm
    cos_sim = f @ f.T
    prob_set = cos_sim / torch.norm(cos_sim, dim=1)
    max_val = 0
    min_val = 1
    max_i, max_j = 0, 0
    min_i, min_j = 1, 1
    min_sim_in_sc = 1
    _i, _j = 0, 0
    max_sim_disjoint_sc = 0
    __i, __j = 0, 0
    with open("./cosine_similarity_matrix.txt", "w") as f:
        for i in range(100):
            for j in range(100):
                val = cos_sim[i,j].item()
                if i != j and val > max_val:
                    max_val = val
                    max_i, max_j = i, j
                elif i != j and val < min_val:
                    min_val = val
                    min_i, min_j = i, j
                _str = f"({val:1.2f}) " if T[i] == T[j] else f" {val:1.2f}  "
                f.write(_str)
                if T[i] == T[j]:
                    if val < min_sim_in_sc:
                        min_sim_in_sc = val
                        _i, _j = i, j
                else:
                    if val > max_sim_disjoint_sc:
                        max_sim_disjoint_sc = val
                        __i, __j = i, j
            f.write("\n")

    with open("./prop_matrix.txt", "w") as f:
        for i in range(100):
            for j in range(100):
                val = prob_set[i,j].item()
                _str = f"({val:1.2f}) " if T[i] == T[j] else f" {val:1.2f}  "
                f.write(_str)
            f.write("\n")

    print("Closest pair of classes:")
    print(f"{set.classes[max_i]} and {set.classes[max_j]} have a cosine similarity of {max_val}\n")

    print("Most distant pair of classes:")
    print(f"{set.classes[min_i]} and {set.classes[min_j]} have a cosine similarity of {min_val}\n")

    print("Most distant pair of classes where both classes are in the same superclass:")
    print(f"{set.classes[_i]} and {set.classes[_j]} have a cosine similarity of {min_sim_in_sc}\n")
    
    print("Closest pair of classes where the classes are in different superclasses:")
    print(f"{set.classes[__i]} and {set.classes[__j]} have a cosine similarity of {max_sim_disjoint_sc}\n")

    cf_matrix = torch.zeros((100,100)).to(DEVICE)
    with torch.no_grad():
        language_features = language_model(language_input)

        for x, t, _ in loader:
            x = x.to(DEVICE)
            sim, _ = model(language_features, x)
            pred = torch.argmax(sim, dim=1)
            N = t.shape[0]
            for i in range(N):
                cf_matrix[t[i]][pred[i]] += 1.0

    cf_matrix_norm = cf_matrix/torch.norm(cf_matrix, dim=1)
    with open("./cf_matrix_norm.txt", "w") as f:
        for i in range(100):
            for j in range(100):
                val = cf_matrix_norm[i,j].item()
                _str = f"({val:1.2f}) " if T[i] == T[j] else f" {val:1.2f}  "
                f.write(_str)
            f.write("\n")
    with open("./cf_matrix.txt", "w") as f:
        for i in range(100):
            for j in range(100):
                val = cf_matrix[i,j].item()
                _str = f"({val:1.2f}) " if T[i] == T[j] else f" {val:1.2f}  "
                f.write(_str)
            f.write("\n")
    
    selected_examples = ['apple', 'pear', 'crab', 'lobster', 'worm', 'snake']
    indexes_selected_examples = [set.classes.index(example) for example in selected_examples]
    with open("./cf_matrix_norm_sel.txt", "w") as f:
        for i in indexes_selected_examples:
            for j in indexes_selected_examples:
                val = cf_matrix_norm[i][j]
                _str = f" {val:1.2f} "
                f.write(_str)
            f.write("\n")

    with open("./prob_sel.txt", "w") as f:
        for i in indexes_selected_examples:
            for j in indexes_selected_examples:
                val = prob_set[i][j]
                _str = f" {val:1.2f} "
                f.write(_str)
            f.write("\n")

get_ang_dist()