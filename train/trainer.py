import torch
import torch.nn as nn
import torch.optim as optim
from utils import DEVICE, Evaluator
from lfm import LocalFeatureMixup


def train(model, train_set, train_loader, validator, loss_fn, epochs, lr, use_lfm, freq, writer):
    optimizer = optim.Adam(model.visual_model.get_parameters(), lr)
    evaluator = Evaluator(train_set.get_class_subdivisions(), loss_fn)
    mixer = LocalFeatureMixup(1, freq)
    with torch.no_grad():
        language_features = model.language_model(train_set.get_lang_inputs())

    def train_step_default(batch):
        x, tgt, _ = batch
        x = x.to(DEVICE)
        tgt = tgt.to(DEVICE)
        optimizer.zero_grad()
        pred, _ = model(language_features, x)
        loss = loss_fn(pred, tgt)
        loss.backward()
        optimizer.step()
        evaluator.update(pred, tgt)

    def train_step_lfm(batch):
        x, y, _ = batch
        x_i, x_j = x
        y_i, y_j = y
        x, y = mixer.mix(x_i, y_i, x_j, y_j)
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pred, _ = model(language_features, x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        evaluator.update(pred, y)
        # exit()

    # report_metrics(None, validator, 0, writer)
    step = 0
    train_step = train_step_default if not use_lfm else train_step_lfm
    for _ in range(epochs):
        for batch in train_loader:
            train_step(batch)
        #     step += 1
        # if use_lfm:
        #     report_metrics(None, validator, step, writer)
        # else:
        #     report_metrics(evaluator, validator, step, writer)

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
    if evaluator:
        # Log training accuracy
        all, many, med, few = evaluator.accuracy()
        training_loss = evaluator.loss()
        writer.add_scalar("Train/Accuracy/All", all.item(), step)
        writer.add_scalar("Train/Accuracy/Many", many.item(), step)
        writer.add_scalar("Train/Accuracy/Med", med.item(), step)
        writer.add_scalar("Train/Accuracy/Few", few.item(), step)
        writer.add_scalar("Train/AvgLoss", training_loss.item(), step)
        evaluator.refresh()
        # Other metrics
        writer.add_scalars(f'1/TrainingAndValidationLosses', {
            'train': training_loss,
            'validation': validator_loss,
        }, step)
 