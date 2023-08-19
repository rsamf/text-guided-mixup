import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import Evaluator, Validator
from tqdm import tqdm

def train(model, device, train_set, train_loader, val_loader, f_l, loss_fn, epochs, lr, mixer, writer, checkpoint=None):
    evaluator = Evaluator(train_set.get_class_subdivisions(), loss_fn, device)
    validator = Validator(model, f_l, val_loader, train_set.get_class_subdivisions(), loss_fn, device)

    def report_metrics(step, only_validate=False):
        # Log validation accuracy
        validator.evaluate()
        all, many, med, few = validator.accuracy()
        validator_loss = validator.loss()
        writer.add_scalar("Validation/Accuracy/All", all.item(), step)
        writer.add_scalar("Validation/Accuracy/Many", many.item(), step)
        if med != None:
            writer.add_scalar("Validation/Accuracy/Med", med.item(), step)
        if few != None:
            writer.add_scalar("Validation/Accuracy/Few", few.item(), step)
        writer.add_scalar("Validation/AvgLoss", validator_loss.item(), step)
        if not only_validate:
            # training_loss = evaluator.loss(use_one_hot=mixer!=None)
            # writer.add_scalar("Train/AvgLoss", training_loss.item(), step)
            evaluator.refresh()

    def train_step_default(batch, optimizer, phase):
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred, _ = model(f_l, x, phase)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        evaluator.update(pred, y)

    def train_step_mixup(batch, optimizer, phase):
        x, y, _ = batch
        x, y_a, y_b, lam = mixer.mix(x, y)
        x = x.to(device)
        y = y.to(device)
        y_hot = F.one_hot(y, freq.shape[0]).to(torch.float)
        optimizer.zero_grad()
        pred, _ = model(f_l, x, phase)
        loss = mixer.mixup_criterion(loss_fn, pred, y_a, y_b, lam)
        loss.backward()
        optimizer.step()
        evaluator.update(pred, None, y_hot)

    def train_step_remix(batch, optimizer, phase):
        x, y, _ = batch
        x, y = mixer.mix(x, y)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred, _ = model(f_l, x, phase)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        evaluator.update(pred, None, y)

    def train_step_lfm(batch, optimizer, phase):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred, _ = model(f_l, x, phase)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        del loss, pred

    step = 0
    optimizers = [optim.Adam(model.clip_params(), lr[0]), optim.Adam(model.fc_params(), lr[1])]
    if mixer == None:
        train_steps = [train_step_default, train_step_default]
    else:
        train_steps = [train_step_lfm, train_step_lfm]
    phase_start = 0
    epoch_start = 0
    if checkpoint != None:
        print("Loading checkpoint")
        model_state = torch.load(checkpoint)
        phase_start = 1
        epoch_start = 0
        model.load_state_dict(model_state)
    for phase in range(phase_start, 2):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizers[phase], epochs[phase], eta_min=lr[phase]/1000)
        report_metrics(step, only_validate=True)
        for i in range(epoch_start, epochs[phase]):
            print(f"Phase {phase}, Epoch {i}")
            model.train()
            for batch in tqdm(train_loader[phase]):
                train_steps[phase](batch, optimizers[phase], phase)
                step += 1
            report_metrics(step, only_validate=False)
            scheduler.step()
        torch.save(model.state_dict(), f"decoupled_single_gpu_{phase}_{device}.pt")
