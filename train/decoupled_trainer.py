import torch
import torch.optim as optim
from utils import DEVICE, Evaluator
from lfm import LocalFeatureMixup

def train(model, train_set, train_loader, validator, loss_fn, epochs, lr, use_lfm, alpha, freq, writer, phase1_model=None):
    torch.cuda.set_per_process_memory_fraction(.75)
    evaluator = Evaluator(train_set.get_class_subdivisions(), loss_fn)
    mixer = LocalFeatureMixup(alpha, freq)
    with torch.no_grad():
        language_features = model.get_text_features(train_set.get_lang_inputs())

    def train_step_default(batch, optimizer, phase=0):
        x, tgt, _ = batch
        x = x.to(DEVICE)
        tgt = tgt.to(DEVICE)
        optimizer.zero_grad()
        pred, _ = model(language_features, x, phase)
        loss = loss_fn(pred, tgt)
        loss.backward()
        optimizer.step()
        evaluator.update(pred, tgt)

    def train_step_lfm(batch, optimizer, phase=0):
        x, y, _ = batch
        x_i, x_j = x
        y_i, y_j = y
        x, y = mixer.mix(x_i, y_i, x_j, y_j)
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pred, _ = model(language_features, x, phase)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        evaluator.update(pred, y) 

    report_metrics(None, validator, 0, writer)
    step = 0
    train_step = train_step_default if not use_lfm else train_step_lfm
    # train_step = train_step_lfm_features
    for phase in range(2):
        if phase == 0:
            if phase1_model != None:
                model.load_state_dict(torch.load(phase1_model))
                continue
            optimizer = optim.Adam(model.phase0_params, lr[0])
        else:
            optimizer = optim.Adam(model.phase1_params, lr[1])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs[phase], eta_min=lr[phase]/1000) #eta_min=1e-7
        for i in range(epochs[phase]):
            writer.add_scalar("Misc/LR", scheduler.get_last_lr()[0], step)
            print(f"Phase {phase}, Epoch {i}")
            model.train()
            for batch in train_loader:
                train_step(batch, optimizer, phase)
                step += 1
            report_metrics(evaluator, validator, step, writer)
            scheduler.step()
        torch.save(model.state_dict(), f"decoupled_trained_phase_{phase}.pt")
        torch.cuda.empty_cache()

def report_metrics(evaluator, validator, step, writer):
    # Log validation accuracy
    validator.evaluate()
    all, many, med, few = validator.accuracy()
    validator_loss = validator.loss()
    writer.add_scalar("Validation/Accuracy/All", all.item(), step)
    writer.add_scalar("Validation/Accuracy/Many", many.item(), step)
    writer.add_scalar("Validation/Accuracy/Med", med.item(), step)
    writer.add_scalar("Validation/Accuracy/Few", few.item(), step)
    writer.add_scalar("Validation/AvgLoss", validator_loss.item(), step)
    if evaluator != None:
        # Log training accuracy
        # all, many, med, few = evaluator.accuracy()
        # training_loss = evaluator.loss()
        # writer.add_scalar("Train/Accuracy/All", all.item(), step)
        # writer.add_scalar("Train/Accuracy/Many", many.item(), step)
        # writer.add_scalar("Train/Accuracy/Med", med.item(), step)
        # writer.add_scalar("Train/Accuracy/Few", few.item(), step)
        # writer.add_scalar("Train/AvgLoss", training_loss.item(), step)
        # Other metrics
        # writer.add_scalars(f'1/TrainingAndValidationLosses', {
        #     'train': training_loss,
        #     'validation': validator_loss,
        # }, step)
        many, med, few = evaluator.observed_labels()
        writer.add_scalar("Misc/Labels/Many", many.item(), step)
        writer.add_scalar("Misc/Labels/Med", med.item(), step)
        writer.add_scalar("Misc/Labels/Few", few.item(), step)
        evaluator.refresh()
 