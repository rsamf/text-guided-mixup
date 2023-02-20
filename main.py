import argparse
import torch
from data import dataloader
from losses import AFS, BalancedCE
from models.simple import SimpleCLIPModel
import torch.optim as optim
from utils import DEVICE
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


data_root = {'ImageNet': './dataset/ImageNet',
             'Places': './dataset/Places-LT',
             'iNaturalist18': '/checkpoint/bykang/iNaturalist18',
             'CIFAR10': './dataset/CIFAR10',
             'CIFAR100': './dataset/CIFAR100',
             }

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--save_feat', type=str, default='')

args = parser.parse_args()


writer = SummaryWriter()

def main():
    model = SimpleCLIPModel(100).to(DEVICE)
    dr = data_root['CIFAR100']
    train_set, train_loader = dataloader.load_data(dr, 'CIFAR100_LT', 'train', 4, num_workers=4, shuffle=True, cifar_imb_ratio=100, transform=model.preprocess)
    val_set, val_loader = dataloader.load_data(dr, 'CIFAR100_LT', 'val', 4, num_workers=4, shuffle=True, transform=model.preprocess)
    train_stage_1(model, train_set)
    train_stage_2(train_set, train_loader, model, train_set.get_super_class_mapping())
    train_stage_2(train_set, train_loader, model)
    validate(model, val_set, val_loader)


def stage_1_train_step(language_model, language_input, loss_fn, step):
    feature_output = language_model(language_input)
    # weights = language_model.get_parameters()
    loss = loss_fn(feature_output)
    writer.add_scalar("Stage1/Loss", loss, step)
    return loss

def stage_2_train_step(model, language_features, img_input, tgt, loss_fn, step, super_class_mapping=None):
    similarity, f_v = model(language_features, img_input)
    stage_name = "Stage2"
    if super_class_mapping != None:
        similarity = similarity @ super_class_mapping.to(DEVICE)
        stage_name = "SuperClassStage"
    loss = loss_fn(similarity, tgt)
    acc = torch.sum(torch.argmax(similarity, dim=-1) == tgt) / tgt.shape[0]
    writer.add_scalar(f"{stage_name}/Loss", loss, step)
    writer.add_scalar(f"{stage_name}/Accuracy", acc, step)
    return loss

def train_step(optimizer, get_loss, params):
    optimizer.zero_grad()
    loss = get_loss(*params)
    if loss != 0:
        loss.backward()
        optimizer.step()
    return loss

def train_stage_1(model, train_set):
    MAX_ITER = 15000
    language_model = model.language_model
    language_input = train_set.get_lang_inputs()
    stage1_loss = AFS()
    optimizer = optim.Adam(language_model.parameters())
    for i in range(MAX_ITER):
        loss = train_step(optimizer, stage_1_train_step, [language_model, language_input, stage1_loss, i])
        if loss == 0:
            break

def train_stage_2(dataset, train_loader, model, super_class_mapping=(None, None)):
    pred_map_layer, tgt_map = super_class_mapping
    tgt_fn = (lambda y: y) if tgt_map == None else (lambda y: tgt_map[y])
    stage2_loss = BalancedCE(tgt_map)
    with torch.no_grad():
        language_features = model.language_model(dataset.get_lang_inputs())

    optimizer = optim.Adam(model.visual_model.parameters())
    EPOCHS = 10
    step = 0
    for _ in range(EPOCHS):
        for x, tgt, _ in train_loader:
            x = x.to(DEVICE)
            # tgt = tgt.to(device="cuda")
            tgt = torch.stack([tgt_fn(t) for t in tgt]).to(DEVICE)
            train_step(optimizer, stage_2_train_step, [model, language_features, x, tgt, stage2_loss, step, pred_map_layer])
            step += 1

def validate(model, val_set, val_loader):
    with torch.no_grad():
        language_features = model.language_model(val_set.get_lang_inputs())

    all_preds = []
    for x, tgt, _ in val_loader:
        x = x.to(DEVICE)
        tgt = tgt.to(device="cuda")
        similarity = model(language_features, x)
        preds = torch.argmax(similarity, dim=-1) == tgt
        all_preds.append(preds)
    all_preds = torch.stack(all_preds)
    acc = torch.sum(all_preds) / tgt.shape[0]
    print(acc)

main()
