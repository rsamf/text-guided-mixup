import torch
import torch.nn as nn
import torch.linalg as L
from data import dataloader
from models.simple import SimpleCLIPModel
from utils import DEVICE

def get_ang_dist():
    model = SimpleCLIPModel().to(DEVICE)
    dr = './dataset/CIFAR100'
    train_set, train_loader = dataloader.load_data(dr, 'CIFAR100_LT', 'train', 4, num_workers=4, shuffle=True, cifar_imb_ratio=100, transform=model.preprocess)
    language_input = train_set.get_lang_inputs()
    language_model = model.language_model
    S, T = train_set.get_super_class_mapping()
    S = S.to(device=DEVICE, dtype=torch.float)

    with torch.no_grad():
        f = language_model(language_input)

    f_norm = L.vector_norm(f, dim=1, keepdim=True)
    f = f / f_norm
    cos_sim = f @ f.T
    max_val = 0
    min_val = 1
    max_i, max_j = 0, 0
    min_i, min_j = 1, 1
    min_sim_in_sc = 1
    _i, _j = 0, 0
    max_sim_disjoint_sc = 0
    __i, __j = 0, 0
    with open("./torch_tensor.txt", "w") as f:
        # torch.set_printoptions(threshold=10_000)
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
    print(max_val, max_i, max_j)
    print(train_set.classes[max_i], train_set.classes[max_j])

    print(min_val, min_i, min_j)
    print(train_set.classes[min_i], train_set.classes[min_j])

    print(min_sim_in_sc, _i, _j)
    print(train_set.classes[_i], train_set.classes[_j])
    
    print(max_sim_disjoint_sc, __i, __j)
    print(train_set.classes[__i], train_set.classes[__j])
    # S, T = train_set.get_super_class_mapping()
    # S = S.to(device=DEVICE, dtype=torch.float)
    # print(S.shape, f.shape)
    # f_s = S.T @ f
    # print(f_s.shape)

    # cos_sim = f @ f.T
    # max_val = 0
    # min_val = 1
    # max_i, max_j = 0, 0
    # min_i, min_j = 1, 1
    # with open("./torch_tensor_sc.txt", "w") as f:
    #     # torch.set_printoptions(threshold=10_000)
    #     for i in range(20):
    #         for j in range(20):
    #             val = cos_sim[i,j].item()
    #             if i != j and val > max_val:
    #                 max_val = val
    #                 max_i, max_j = i, j
    #             elif i != j and val < min_val:
    #                 min_val = val
    #                 min_i, min_j = i, j
    #             _str = f"{val:1.2f} "
    #             f.write(_str)
    #         f.write("\n")
    # print(max_val, max_i, max_j)
    # print(train_set.classes[max_i], train_set.classes[max_j])

    # print(min_val, min_i, min_j)
    # print(train_set.classes[min_i], train_set.classes[min_j])
    # 
    # print(cos_sim)

get_ang_dist()