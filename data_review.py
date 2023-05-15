import torch
import torch.linalg as L
import torch.nn.functional as F
from data import dataloader
from models.simple import SimpleCLIPModel
from utils import get_sample_probability_matrix_softmax
import os.path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def p(filename):
    return os.path.join('data_review', filename)

def get_ang_dist():
    model = SimpleCLIPModel(device="cpu")
    dr = './dataset/CIFAR100'
    set = dataloader.get_dataset(dr, 'CIFAR100', 'val', model.preprocess)
    loader = dataloader.get_dataloader(set, 32)
    language_input = set.get_lang_inputs()
    language_model = model.get_text_features
    S, T = set.get_super_class_mapping()
    S = S.to(device=DEVICE, dtype=torch.float)

    with torch.no_grad():
        f = language_model(language_input)

    f_norm = L.vector_norm(f, dim=1, keepdim=True)
    f = f / f_norm
    cos_sim = f @ f.T
    prob_set = get_sample_probability_matrix_softmax(language_model, language_input, .05, set.classes)
    max_val = 0
    min_val = 1
    max_i, max_j = 0, 0
    min_i, min_j = 1, 1
    min_sim_in_sc = 1
    _i, _j = 0, 0
    max_sim_disjoint_sc = 0
    __i, __j = 0, 0
    with open(p("./cosine_similarity_matrix.txt"), "w") as f:
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

    with open(p("./prop_matrix.txt"), "w") as f:
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

    model = SimpleCLIPModel(device=DEVICE).to(DEVICE)
    language_model = model.get_text_features
    cf_matrix = torch.zeros((100,100)).to(DEVICE)

    with torch.no_grad():
        language_features = language_model(language_input)

        for x, t, _ in loader:
            x = x.to(DEVICE)
            sim, _ = model(language_features, x, 1)
            pred = F.softmax(sim, dim=1)
            pred = torch.argmax(pred, dim=1)
            N = t.shape[0]
            for i in range(N):
                cf_matrix[t[i]][pred[i]] += 1.0

    with open(p("./cf_matrix.txt"), "w") as f:
        for i in range(100):
            for j in range(100):
                val = cf_matrix[i,j].item()
                _str = f"({val:1.2f}) " if T[i] == T[j] else f" {val:1.2f}  "
                f.write(_str)
            f.write("\n")
    
    selected_examples = ['apple', 'pear', 'crab', 'lobster', 'worm', 'snake']
    indexes_selected_examples = [set.classes.index(example) for example in selected_examples]
    with open(p("./cf_matrix_sel.txt"), "w") as f:
        for i in indexes_selected_examples:
            for j in indexes_selected_examples:
                val = cf_matrix[i][j]
                _str = f" {val:1.2f} "
                f.write(_str)
            f.write("\n")

    with open(p("./prob_sel.txt"), "w") as f:
        for i in indexes_selected_examples:
            for j in indexes_selected_examples:
                val = prob_set[i][j]
                _str = f" {val:1.5f} "
                f.write(_str)
            f.write("\n")

get_ang_dist()