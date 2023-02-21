import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Validator():
    def __init__(self, model, val_set, val_loader, class_subdivisions):
        self.model = model
        self.val_set = val_set
        self.val_loader = val_loader
        self.class_subdivisions = class_subdivisions

    def accuracy(self):
        with torch.no_grad():
            language_features = self.model.language_model(self.val_set.get_lang_inputs())
            evaluator = Evaluator(self.class_subdivisions)
            for x, tgt, _ in self.val_loader:
                x = x.to(DEVICE)
                tgt = tgt.to(DEVICE)
                logits, _ = self.model(language_features, x)
                evaluator.update(logits, tgt)
            return evaluator.accuracy()

class Evaluator():
    def __init__(self, class_subdivisions):
        self.logits = []
        self.tgts = []
        self.class_subdivisions = class_subdivisions

    def update(self, logits, tgts):
        self.logits.append(logits)
        self.tgts.append(tgts)

    def get_tensors(self):
        return torch.cat(self.logits), torch.cat(self.tgts)

    def refresh(self):
        self.logits = []
        self.tgts = []

    def accuracy(self):
        def acc(logits, tgts):
            if tgts.shape[0] == 0:
                return torch.zeros(1)
            return torch.sum(torch.argmax(logits, dim=-1) == tgts) / tgts.shape[0]
        logits, tgts = self.get_tensors()
        all_l, all_t = [], []
        many_l, many_t = [], []
        med_l, med_t = [], []
        few_l, few_t = [], []

        for i in range(tgts.shape[0]):
            l, t = logits[i], tgts[i]
            all_l.append(l)
            all_t.append(t)
            if self.class_subdivisions[t] == "many":
                many_l.append(l)
                many_t.append(t)
            elif self.class_subdivisions[t] == "med":
                med_l.append(l)
                med_t.append(t)
            else:
                few_l.append(l)
                few_t.append(t)

        all_l, all_t = torch.stack(all_l), torch.stack(all_t)
        many_l, many_t = torch.stack(many_l), torch.stack(many_t)
        med_l, med_t = torch.stack(med_l), torch.stack(med_t)
        few_l, few_t = torch.stack(few_l), torch.stack(few_t)
        return acc(all_l, all_t), acc(many_l, many_t), acc(med_l, med_t), acc(few_l, few_t)
            
