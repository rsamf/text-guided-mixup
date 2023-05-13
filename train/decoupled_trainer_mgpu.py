import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import Evaluator, Validator
from mixups import LocalFeatureMixup
import torch.distributed as dist
from tqdm import tqdm

def train(model, device, world_size, train_set, train_loader, val_loader, f_l, loss_fn, epochs, lr, alpha, freq, writer, checkpoint=None, main_device=0):
    evaluator = Evaluator(train_set.get_class_subdivisions(), loss_fn, device, True, world_size)
    validator = Validator(model, f_l, val_loader, train_set.get_class_subdivisions(), loss_fn, device, True, world_size)
    mixer = LocalFeatureMixup(alpha, freq)

    def report_metrics(step, only_validate=False):
        # Log validation accuracy
        validator.evaluate()
        all, many, med, few = validator.accuracy()
        validator_loss = validator.loss()
        if device == main_device:
            writer.add_scalar("Validation/Accuracy/All", all.item(), step)
            writer.add_scalar("Validation/Accuracy/Many", many.item(), step)
            if med != None:
                writer.add_scalar("Validation/Accuracy/Med", med.item(), step)
            if few != None:
                writer.add_scalar("Validation/Accuracy/Few", few.item(), step)
            writer.add_scalar("Validation/AvgLoss", validator_loss.item(), step)
        if not only_validate:
            training_loss = evaluator.loss(use_one_hot=alpha!=None)
            if device == main_device:
                writer.add_scalar("Train/AvgLoss", training_loss.item(), step)
            evaluator.refresh()

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
    optimizers = [optim.Adam(model.module.clip_params(), lr[0]), optim.Adam(model.module.fc_params(), lr[1])]
    train_steps = [train_step_lfm, train_step_lfm]
    phase_start = 0
    epoch_start = 0
    optimizer_state = None
    scheduler_state = None
    if checkpoint != None:
        dist.barrier()
        print("Loading checkpoint")
        map_location = {'cuda:0': 'cuda:%d' % device}
        checkpoint_cfg = torch.load(checkpoint, map_location=map_location)
        phase_start = checkpoint_cfg['phase']
        epoch_start = checkpoint_cfg['epoch'] + 1
        optimizer_state = checkpoint_cfg['optimizer']
        model_state = checkpoint_cfg['model']
        scheduler = checkpoint_cfg['lr_scheduler']
        model.module.load_state_dict(model_state)
    for phase in range(phase_start, 2):
        mixer.set_alpha(phase)
        optimizer = optimizers[phase]
        if optimizer_state != None:
            optimizer.load_state_dict(optimizer_state)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs[phase], eta_min=lr[phase]/1000)
        if scheduler_state != None:
            scheduler.load_state_dict(scheduler_state)
        report_metrics(step, only_validate=True)
        for i in range(epoch_start, epochs[phase]):
            train_loader[phase].sampler.set_epoch(i)
            print(f"Phase {phase}, Epoch {i}")
            model.train()
            for batch in tqdm(train_loader[phase]):
                train_steps[phase](batch, optimizer, phase)
                step += 1
            report_metrics(step, only_validate=False)
            scheduler.step()
            # Saving every epoch
            if device == main_device:
                checkpoint = {
                    'phase': phase,
                    'epoch': i,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'model': model.module.state_dict()
                }
                torch.save(checkpoint, f"mpgpu_{phase}_a.pt")

 