import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

FEAT_DIMS = {
    "RN50": 1024,
    "RN50x16": 768,
    "RN101": 512,
    "ViT-B/32": 512,
    "ViT-B/16": 512
}
class SimpleCLIPModel(nn.Module):
    def __init__(self, device, backbone="RN50"):
        super(SimpleCLIPModel, self).__init__()
        self.device = device
        # Setup encoders
        encoders, self.preprocess = clip.load(backbone, device)
        encoders.to(torch.float)
        feat_dim = FEAT_DIMS[backbone]
        self.encode_text = encoders.encode_text
        # Setup model
        self.visual = encoders.visual
        self.fc = nn.Linear(feat_dim, feat_dim, bias=False)
        nn.init.eye_(self.fc.weight)

    def clip_params(self):
        return self.visual.parameters()
    
    def fc_params(self):
        return self.fc.parameters()

    def get_text_features(self, text_input):
        with torch.no_grad():
            text_input = torch.cat([clip.tokenize(text) for text in text_input]).to(self.device)
            clip_features = self.encode_text(text_input).to(dtype=torch.float)
        return clip_features

    def get_similarity(self, f_v, f_l):
        f_v_norm = f_v.norm(dim=-1, keepdim=True)
        f_l_norm = f_l.norm(dim=-1, keepdim=True)
        similarity = (f_v / f_v_norm) @ (f_l / f_l_norm).T
        return similarity

    def forward(self, f_l, image_input, phase):
        if phase == 0:
            f_v = self.visual(image_input)#.to(dtype=torch.float)
        else:
            with torch.no_grad():
                clip_features = self.visual(image_input)#.to(dtype=torch.float)
            f_v = self.fc(clip_features)
        f_v = f_v.to(torch.float)
        f_v_norm = f_v.norm(dim=-1, keepdim=True)
        similarity = (f_v / f_v_norm) @ f_l.T
        return similarity, f_v
