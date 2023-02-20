import os
import torch
import torch.nn as nn
import torch.linalg as lin
import clip
from utils import DEVICE

class LanguageModel(nn.Module):
    def __init__(self, num_classes, clip_encode):
        super(LanguageModel, self).__init__()
        self.clip_encode = clip_encode
        # self.fc = nn.Linear(512, 512)

    def get_parameters(self):
        return self.fc.weight.data

    def forward(self, text_input):
        text_input = torch.cat([clip.tokenize(text) for text in text_input]).to(DEVICE)
        with torch.no_grad():
            clip_features = self.clip_encode(text_input).to(dtype=torch.float)
        # mapped_features = self.fc(clip_features)
        return clip_features

class VisualModel(nn.Module):
    def __init__(self, num_classes, clip_encode):
        super(VisualModel, self).__init__()
        self.clip_encode = clip_encode
        self.fc = nn.Linear(512, 512)
    
    def get_parameters(self):
        return self.fc.weight.data

    def forward(self, image_input):
        with torch.no_grad():
            clip_features = self.clip_encode(image_input).to(dtype=torch.float)
        mapped_features = self.fc(clip_features)
        return mapped_features

class SimpleCLIPModel(nn.Module):
    def __init__(self, num_classes, backbone="ViT-B/32"):
        super(SimpleCLIPModel, self).__init__()
        encoders, self.preprocess = clip.load(backbone, DEVICE)
        self.language_model = LanguageModel(num_classes, encoders.encode_text).to(DEVICE)
        self.visual_model = VisualModel(num_classes, encoders.encode_image).to(DEVICE)
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, text_features, image_input):
        f_v = self.visual_model(image_input)
        f_l = text_features

        f_v_norm = f_v.norm(dim=-1, keepdim=True)
        f_l_norm = f_l.norm(dim=-1, keepdim=True)
        similarity = (f_v / f_v_norm) @ (f_l / f_l_norm).T
        return similarity, f_v
