import pickle
import torch


def get_optimizer(optmizer_type, model, lr=0.1):
    # Funcion para rehacer el optmizador -> Ayuda para cambiar learning rate
    if optmizer_type == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    if optmizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optmizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optmizer_type == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)

    assert False, 'No optimizers with that name! [' + optmizer_type + ']'


def get_scheduler(optimizer, steps, plateau):
    if steps:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 130, 170], gamma=0.1)
    elif plateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=6, factor=0.1, patience=12)
    else:
        return None


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_progress(progress_train_loss, progress_val_loss, progress_train_accuracy, progress_val_accuracy, model,
                  output_dir):
    torch.save(model.state_dict(), output_dir + "/model_last.pt")

    progress = {"train_loss": progress_train_loss, "val_loss": progress_val_loss,
                "train_accuracy": progress_train_accuracy, "val_accuracy": progress_val_accuracy}

    with open(output_dir + 'progress.pickle', 'wb') as handle:
        pickle.dump(progress, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_step(train_loader, model, criterion, optimizer):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = (train_loss / (batch_idx + 1))
    train_accuracy = 100. * correct / total
    return train_loss, train_accuracy


def val_step(val_loader, model, criterion):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        val_loss = (val_loss / (batch_idx + 1))
        val_accuracy = 100. * correct / total
    return val_loss, val_accuracy
