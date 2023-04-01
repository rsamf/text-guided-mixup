import torch
import torch.nn as nn
import torch.optim as optim
from utils import DEVICE, Evaluator

def train(model, train_set, train_loader, validator, loss_fn, epochs, lr, writer):
    with torch.no_grad():
        language_features = model.language_model(train_set.get_lang_inputs())

    optimizer = optim.Adam(model.visual_model.get_parameters(), lr)
    evaluator = Evaluator(train_set.get_class_subdivisions(), loss_fn)
    report_metrics(None, validator, 0, writer)
    step = 0
    for _ in range(epochs):
        for x, tgt, _ in train_loader:
            step += 1
            x = x.to(DEVICE)
            tgt = tgt.to(DEVICE)
            optimizer.zero_grad()
            pred, _ = model(language_features, x)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()
            evaluator.update(pred, tgt)
        report_metrics(evaluator, validator, step, writer)

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
 