import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import Evaluator, Validator
from lfm import LocalFeatureMixup

def train(model, device, train_set, train_loader, val_loader, f_l, loss_fn, epochs, lr, alpha, freq, writer, phase1_model=None):
    torch.cuda.set_per_process_memory_fraction(.75)
    evaluator = Evaluator(train_set.get_class_subdivisions(), loss_fn, device)
    validator = Validator(model, f_l, val_loader, train_set.get_class_subdivisions(), loss_fn, device)
    mixer = LocalFeatureMixup(alpha, freq)

    def report_metrics(step, only_validate=False):
        # Log validation accuracy
        validator.evaluate()
        all, many, med, few = validator.accuracy()
        validator_loss = validator.loss()
        writer.add_scalar("Validation/Accuracy/All", all.item(), step)
        writer.add_scalar("Validation/Accuracy/Many", many.item(), step)
        writer.add_scalar("Validation/Accuracy/Med", med.item(), step)
        writer.add_scalar("Validation/Accuracy/Few", few.item(), step)
        writer.add_scalar("Validation/AvgLoss", validator_loss.item(), step)
        if not only_validate:
            training_loss = evaluator.loss()
            writer.add_scalar("Train/AvgLoss", training_loss.item(), step)
            evaluator.refresh()

    def train_step_default(batch, optimizer, phase):
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        y_hot = F.one_hot(y, freq.shape[0]).to(torch.float)
        optimizer.zero_grad()
        pred, _ = model(f_l, x, phase)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        evaluator.update(pred, y, y_hot)

    def train_step_lfm(batch, optimizer, phase):
        x, y, _ = batch
        x_i, x_j = x
        y_i, y_j = y
        x, y, y_no_offset = mixer.mix(x_i, y_i, x_j, y_j)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred, _ = model(f_l, x, phase)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        y_no_offset.to(device)
        evaluator.update(pred, None, y, y_no_offset) 

    step = 0
    optimizers = [optim.Adam(model.clip_params(), lr[0]), optim.Adam(model.fc_params(), lr[1])]
    train_steps = [train_step_lfm, train_step_lfm]
    for phase in range(2):
        if phase == 0 and phase1_model != None:
            map_location = {'cuda:0': 'cuda:%d' % device}
            model.load_state_dict(torch.load(phase1_model, map_location=map_location))
            continue
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizers[phase], epochs[phase], eta_min=lr[phase]/1000)
        report_metrics(step, only_validate=True)
        for i in range(epochs[phase]):
            print(f"Phase {phase}, Epoch {i}")
            model.train()
            for batch in train_loader[phase]:
                train_steps[phase](batch, optimizers[phase], phase)
                step += 1
            report_metrics(step, only_validate=False)
            scheduler.step()
        torch.save(model.state_dict(), f"decoupled_single_gpu_{phase}.pt")
