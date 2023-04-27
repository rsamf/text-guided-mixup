import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from utils import DEVICE
from collections import OrderedDict

class LanguageModel(nn.Module):
    def __init__(self, clip_encode):
        super(LanguageModel, self).__init__()
        self.clip_encode = clip_encode

    def forward(self, text_input):
        text_input = torch.cat([clip.tokenize(text) for text in text_input]).to(DEVICE)
        with torch.no_grad():
            clip_features = self.clip_encode(text_input).to(dtype=torch.float)
        return clip_features

class VisualModel(nn.Module):
    def __init__(self, clip_encode):
        super(VisualModel, self).__init__()
        self.clip_encode = clip_encode
        self.fc = nn.Linear(512, 512, bias=False)
        nn.init.eye_(self.fc.weight)
    
    def get_parameters(self):
        return self.fc.parameters()

    def forward(self, image_input):
        with torch.no_grad():
            clip_features = self.clip_encode(image_input).to(dtype=torch.float)
        mapped_features = self.fc(clip_features)
        return mapped_features

class VisualModelV2(nn.Module):
    def __init__(self, clip_encode, num_out_layers=3):
        super(VisualModelV2, self).__init__()
        self.clip_encode = clip_encode
        self.out_layers = nn.Sequential(self.get_out_layers(num_out_layers))

    def get_out_layers(self, num_layers):
        layers = []
        for i in range(num_layers - 1):
            l = nn.Linear(512, 512)
            nn.init.xavier_normal_(l.weight)
            layers.append((f'linear{i}', l))
            r = nn.ReLU()
            layers.append((f'relu{i}', r))
        final = nn.Linear(512, 512)
        nn.init.xavier_normal_(final.weight)
        layers.append(('final_linear', final))
        return OrderedDict(layers)

    def get_parameters(self):
        return self.out_layers.parameters()

    def forward(self, image_input):
        with torch.no_grad():
            clip_features = self.clip_encode(image_input).to(dtype=torch.float)
        mapped_features = self.out_layers(clip_features)
        return mapped_features

class SimpleCLIPModel(nn.Module):
    def __init__(self, backbone="ViT-B/32"):
        super(SimpleCLIPModel, self).__init__()
        encoders, self.preprocess = clip.load(backbone, DEVICE)
        self.language_model = LanguageModel(encoders.encode_text).to(DEVICE)
        self.visual_model = VisualModel(encoders.encode_image).to(DEVICE)
        self.cos = nn.CosineSimilarity(dim=-1)

    def get_similarity(self, f_v, f_l):
        f_v_norm = f_v.norm(dim=-1, keepdim=True)
        f_l_norm = f_l.norm(dim=-1, keepdim=True)
        similarity = (f_v / f_v_norm) @ (f_l / f_l_norm).T
        return similarity

    def forward(self, text_features, image_input):
        f_v = self.visual_model(image_input)
        f_l = text_features

        f_v_norm = f_v.norm(dim=-1, keepdim=True)
        f_l_norm = f_l.norm(dim=-1, keepdim=True)
        similarity = (f_v / f_v_norm) @ (f_l / f_l_norm).T
        return similarity, f_v
