import torch
import torch.linalg as L
import torch.nn.functional as F
import torch.distributed as dist

class Validator():
    def __init__(self, model, f_l, val_loader, class_subdivisions, loss_fn, mgpu=False, world_size=0):
        self.model = model
        self.val_loader = val_loader
        self.class_subdivisions = class_subdivisions
        self.evaluator = Evaluator(class_subdivisions, loss_fn, mgpu, world_size)
        self.f_l = f_l
        self.mgpu = mgpu
        self.world_size = world_size

    def accuracy(self):
        return self.evaluator.accuracy()

    def loss(self):
        return self.evaluator.loss()

    def evaluate(self):
        self.evaluator.refresh()
        self.model.eval()
        with torch.no_grad():
            for x, tgt, _ in self.val_loader:
                x = x.to(self.device)
                y = tgt.to(self.device)
                tgt = F.one_hot(y, num_classes=self.f_l.shape[0]).to(torch.float)
                logits, _ = self.model(self.f_l, x, 1)
                self.evaluator.update(logits, y, tgt)

class Evaluator():
    def __init__(self, class_subdivisions, loss_fn, device, mpgu=False, world_size=0):
        self.logits = []
        self.tgts = []
        self.tgts_one_hot = []
        self.tgts_no_offset = []
        self.class_subdivisions = class_subdivisions
        self.loss_fn = loss_fn
        self.device = device
        self.mpgu = mpgu
        self.world_size = world_size
        self.many_mask = torch.tensor([1. if c == "many" else 0. for c in self.class_subdivisions]).to(self.device)
        self.med_mask = torch.tensor([1. if c == "med" else 0. for c in self.class_subdivisions]).to(self.device)
        self.few_mask = torch.tensor([1. if c == "few" else 0. for c in self.class_subdivisions]).to(self.device)

    def update(self, logits, tgts, tgts_one_hot, tgts_no_offset=None):
        self.logits.append(logits)
        self.tgts.append(tgts)
        self.tgts_one_hot.append(tgts_one_hot)
        if tgts_no_offset != None:
            self.tgts_no_offset.append(tgts_no_offset)

    def get_tensors(self):
        logits, tgts = torch.cat(self.logits).to(self.device), torch.cat(self.tgts).to(self.device)
        if self.mpgu:
            logits_out = [torch.zeros_as(logits, device=self.device) for _ in range(self.world_size)]
            tgts_out = [torch.zeros_as(tgts, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(logits_out, logits)
            dist.all_gather(tgts_out, tgts)
            return logits_out, tgts_out
        return logits, tgts

    def get_tensors_one_hot(self):
        logits, tgts = torch.cat(self.logits).to(self.device), torch.cat(self.tgts_one_hot).to(self.device)
        if self.mpgu:
            logits_out = [torch.zeros_as(logits, device=self.device) for _ in range(self.world_size)]
            tgts_out = [torch.zeros_as(tgts, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(logits_out, logits)
            dist.all_gather(tgts_out, tgts)
            return logits_out, tgts_out
        return logits, tgts
        
    # Doesn't work on mgpu
    def get_observed_labels(self):
        input = torch.cat(self.tgts_no_offset).to(self.device)
        adjusted = torch.cat(self.tgts_one_hot).to(self.device)
        return input, adjusted

    def refresh(self):
        self.logits = []
        self.tgts = []
        self.tgts_one_hot = []
        self.tgts_no_offset = []
    
    def loss(self):
        with torch.no_grad():
            logits, tgts = self.get_tensors_one_hot()
            if self.device == 0:
                if tgts.shape[0] == 0:
                    return torch.zeros(1)
                return self.loss_fn(logits, tgts)
            return None
    
    def observed_labels(self):
        with torch.no_grad():
            input, adjusted = self.get_observed_labels()
            i_cls_freq = torch.sum(input, dim=0)
            a_cls_freq = torch.sum(adjusted, dim=0)
            i_many, i_med, i_few = self.many_mask @ i_cls_freq, self.med_mask @ i_cls_freq, self.few_mask @ i_cls_freq
            a_many, a_med, a_few = self.many_mask @ a_cls_freq, self.med_mask @ a_cls_freq, self.few_mask @ a_cls_freq
            return (i_many, i_med, i_few), (a_many, a_med, a_few)

    def accuracy(self):
        with torch.no_grad():
            def acc(logits, tgts):
                if tgts.shape[0] == 0:
                    return torch.zeros(1)
                return torch.sum(torch.argmax(logits.softmax(dim=-1), dim=-1) == tgts) / tgts.shape[0]
            logits, tgts = self.get_tensors()
            if self.device != 0:
                return None, None, None, None
            
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


def get_sample_probability_matrix_norm(language_model, language_input):
    with torch.no_grad():
        f = language_model(language_input)

    f_norm = L.vector_norm(f, dim=1, keepdim=True)
    f = f / f_norm
    cos_sim = f @ f.T
    I_d = torch.eye(cos_sim.shape[0])
    prob_set = (1 - I_d) * cos_sim
    div = torch.sum(prob_set, dim=1, keepdim=True)
    prob_set = prob_set / div

    return prob_set.to(device='cpu')

def get_sample_probability_matrix_softmax(language_model, language_input, class_list=None, top_k=0):
    with torch.no_grad():
        f = language_model(language_input)

    f_norm = L.vector_norm(f, dim=1, keepdim=True)
    f = f / f_norm
    cos_sim = f @ f.T
    I_d = torch.eye(cos_sim.shape[0])
    prob_set = ((1 - I_d).T * cos_sim) + (I_d * -1e9)
    prob_set *= 5
    prob_set = F.softmax(prob_set, dim=1)
    if top_k > 0 and class_list != None:
        num_classes = len(class_list)
        _, idx = torch.topk(prob_set, dim=1, k=num_classes-top_k, largest=False)
        for i in range(prob_set.shape[0]):
            prob_set[i] = torch.index_fill(prob_set[i], dim=0, index=idx[i], value=-1e10)
        prob_set = F.softmax(prob_set, dim=1)
    show_closest_to(prob_set, class_list, 6)

    return prob_set.to(device='cpu')

def show_closest_to(prob_set, class_list, top_k=6):
    val, idx = torch.topk(prob_set, k=top_k)
    for i in range(10):
        to_print = class_list[i]
        to_print += ": "
        for j in range(top_k):
            to_print += f"({class_list[idx[i][j]]}, {val[i][j]:.5f}) "
        print(to_print)

def get_text_distances(language_model, language_input):
    with torch.no_grad():
        f = language_model(language_input)

    f_norm = L.vector_norm(f, dim=1, keepdim=True)
    f = f / f_norm
    cos_sim = f @ f.T
    dist = 1-cos_sim
    return dist
