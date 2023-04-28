import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from utils import DEVICE
from collections import OrderedDict

class VisualModel(nn.Module):
    def __init__(self, encode_image, feat_dim):
        super(VisualModel, self).__init__()
        self.encode_image = encode_image
        self.fc = nn.Linear(feat_dim, feat_dim, bias=False)
        nn.init.eye_(self.fc.weight)

    def forward(self, image_input, phase=0):
        if phase == 0:
            clip_features = self.encode_image(image_input).to(dtype=torch.float)
            return clip_features
        else:
            with torch.no_grad():
                clip_features = self.encode_image(image_input).to(dtype=torch.float)
            mapped_features = self.fc(clip_features)
            return mapped_features

# class VisualModelV2(nn.Module):
#     def __init__(self, clip_encode, num_out_layers=3):
#         super(VisualModelV2, self).__init__()
#         self.clip_encode = clip_encode
#         self.out_layers = nn.Sequential(self.get_out_layers(num_out_layers))

#     def get_out_layers(self, num_layers):
#         layers = []
#         for i in range(num_layers - 1):
#             l = nn.Linear(512, 512)
#             nn.init.xavier_normal_(l.weight)
#             layers.append((f'linear{i}', l))
#             r = nn.ReLU()
#             layers.append((f'relu{i}', r))
#         final = nn.Linear(512, 512)
#         nn.init.xavier_normal_(final.weight)
#         layers.append(('final_linear', final))
#         return OrderedDict(layers)

#     def get_parameters(self):
#         return self.out_layers.parameters()

#     def forward(self, image_input):
#         with torch.no_grad():
#             clip_features = self.clip_encode(image_input).to(dtype=torch.float)
#         mapped_features = self.out_layers(clip_features)
#         return mapped_features

FEAT_DIMS = {
    "RN50": 1024,
    "RN50x16": 768,
    "RN101": 512,
    "ViT-B/32": 512,
    "ViT-B/16": 512
}
class SimpleCLIPModel(nn.Module):
    def __init__(self, backbone="RN50"):
        super(SimpleCLIPModel, self).__init__()
        encoders, self.preprocess = clip.load(backbone, DEVICE)
        encoders.to(torch.float)
        self.encode_text = encoders.encode_text
        self.encode_image = encoders.encode_image
        feat_dim = FEAT_DIMS[backbone]
        self.visual = VisualModel(self.encode_image, feat_dim).to(DEVICE)
        self.phase0_params = encoders.visual.parameters()
        self.phase1_params = self.visual.fc.parameters()
        self._v = encoders.visual

    def train(self):
        self._v.train()

    def eval(self):
        self._v.eval()

    def get_text_features(self, text_input):
        with torch.no_grad():
            text_input = torch.cat([clip.tokenize(text) for text in text_input]).to(DEVICE)
            clip_features = self.encode_text(text_input).to(dtype=torch.float)
        return clip_features

    def get_similarity(self, f_v, f_l):
        f_v_norm = f_v.norm(dim=-1, keepdim=True)
        f_l_norm = f_l.norm(dim=-1, keepdim=True)
        similarity = (f_v / f_v_norm) @ (f_l / f_l_norm).T
        return similarity

    def forward(self, text_features, image_input, phase=0):
        f_v = self.visual(image_input, phase).to(torch.float)
        f_l = text_features
        f_v_norm = f_v.norm(dim=-1, keepdim=True)
        f_l_norm = f_l.norm(dim=-1, keepdim=True)
        similarity = (f_v / f_v_norm) @ (f_l / f_l_norm).T
        return similarity, f_v
