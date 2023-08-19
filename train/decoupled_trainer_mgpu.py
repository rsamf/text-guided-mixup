import torch
import torch.optim as optim
from utils import Evaluator, Validator
import torch.distributed as dist
from tqdm import tqdm

def train(model, device, world_size, train_set, train_loader, val_loader, f_l, loss_fn, epochs, lr, mixer, writer, checkpoint=None, main_device=0):
    evaluator = Evaluator(train_set.get_class_subdivisions(), loss_fn, device, True, world_size)
    validator = Validator(model, f_l, val_loader, train_set.get_class_subdivisions(), loss_fn, device, True, world_size)

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
            training_loss = evaluator.loss(use_one_hot=mixer!=None)
            if device == main_device:
                writer.add_scalar("Train/AvgLoss", training_loss.item(), step)
            evaluator.refresh()

    def train_step(batch, optimizer, phase):
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
    optimizers = [optim.Adam(model.module.clip_params(), lr[0]), optim.Adam(model.module.fc_params(), lr[1])]
    train_steps = [train_step, train_step]
    phase_start = 0
    epoch_start = 0
    optimizer_state = None
    scheduler_state = None
    if checkpoint != None:
        dist.barrier()
        print("Loading checkpoint")
        map_location = {'cuda:0': 'cuda:%d' % device}
        checkpoint_cfg = torch.load(checkpoint, map_location=map_location)
        model_state = checkpoint_cfg['model']
        # model_state['fc.bias'] = torch.zeros(1024)
        model.module.load_state_dict(model_state)
        phase_start = 1
    for phase in range(phase_start, 2):
        mixer.set_phase(phase)
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
            report_metrics(step, only_validate=True)
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
                torch.save(checkpoint, f"mpgpu_{phase}.pt")

 