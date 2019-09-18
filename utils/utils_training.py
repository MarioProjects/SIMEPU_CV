import pickle
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans


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


def train_analysis(model, val_loader, output_dir, LABELS2TARGETS, TARGETS2LABELS):
    """
    Generate accuracy per class and confusion matrix plots
    """
    class_correct = list(0. for i in range(len(LABELS2TARGETS)))
    class_total = list(0. for i in range(len(LABELS2TARGETS)))
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).squeeze()  # lista con los labels correctos (1) e incorrectos (0)
            # Recorro todos mis labels del batch actual
            for i in range(len(labels)):
                label = labels[i]  # Extraigo el label en concreto del bucle
                class_correct[label] += correct[i].item()  # Si el label ha sido clasificado bien sumaremos 1
                class_total[label] += 1  # Sumo 1 al contador del label actual
    assert len(val_loader.dataset) == np.array(class_total).sum(), "Not processed all?!"

    accuracy_per_class = 100 * np.array(class_correct) / np.array(class_total)

    for i in range(len(LABELS2TARGETS)):
        print('Accuracy of {} : {:.2f}% '.format(TARGETS2LABELS[i], accuracy_per_class[i]))

    sns.set(style="whitegrid")
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax = sns.barplot(ax=ax, x=np.arange(len(LABELS2TARGETS)), y=accuracy_per_class)
    plt.xticks(np.arange(len(LABELS2TARGETS)), list(LABELS2TARGETS.keys()))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75)
    plt.xlabel("Clase")
    plt.ylabel("Accuracy (%)")

    axes = plt.gca()
    axes.set_ylim([0, 100])
    plt.yticks(np.arange(0, 101, 10.0))

    plt.savefig("{}/accuracy_per_class.jpg".format(output_dir), bbox_inches="tight")

    nb_classes = 9

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(val_loader):
            inputs, classes = inputs.cuda(), classes.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print(confusion_matrix)

    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    # Great cmaps -> YlGnBu / GnBu
    ax = sns.heatmap(confusion_matrix.data.cpu().numpy() / np.array(class_total), cmap="YlGnBu", annot=True,
                     linewidths=.5)
    ax.set_yticklabels(ax.get_xticklabels(), rotation=0)
    plt.yticks(np.arange(len(LABELS2TARGETS)), list(LABELS2TARGETS.keys()))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.xticks(np.arange(len(LABELS2TARGETS)), list(LABELS2TARGETS.keys()))

    trans = mtrans.Affine2D().translate(30, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() + trans)

    trans = mtrans.Affine2D().translate(0, -25)
    for t in ax.get_yticklabels():
        t.set_transform(t.get_transform() + trans)

    plt.savefig("{}/confusion_matrix.jpg".format(output_dir), bbox_inches="tight")