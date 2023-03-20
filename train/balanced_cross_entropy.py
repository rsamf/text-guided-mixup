import torch
import torch.nn as nn
import torch.optim as optim
from losses import BalancedCE
from utils import DEVICE, writer, Evaluator

def train(model, train_set, train_loader, validator):
    loss_fn = BalancedCE()
    with torch.no_grad():
        language_features = model.language_model(train_set.get_lang_inputs())

    optimizer = optim.Adam(model.visual_model.get_parameters())
    evaluator = Evaluator(train_set.get_class_subdivisions())
    report_accuracy(evaluator, validator, 0)
    EPOCHS = 10
    step = 0
    for _ in range(EPOCHS):
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
            writer.add_scalar("Train/Loss", loss.item(), step)
        report_accuracy(evaluator, validator, step)

def report_accuracy(evaluator, validator, step):
    # Log training accuracy
    all, many, med, few = evaluator.accuracy()
    writer.add_scalar("Train/Accuracy/All", all.item(), step)
    writer.add_scalar("Train/Accuracy/Many", many.item(), step)
    writer.add_scalar("Train/Accuracy/Med", med.item(), step)
    writer.add_scalar("Train/Accuracy/Few", few.item(), step)
    # Log validation accuracy
    all, many, med, few = validator.accuracy()
    writer.add_scalar("Validation/Accuracy/All", all.item(), step)
    writer.add_scalar("Validation/Accuracy/Many", many.item(), step)
    writer.add_scalar("Validation/Accuracy/Med", med.item(), step)
    writer.add_scalar("Validation/Accuracy/Few", few.item(), step)
