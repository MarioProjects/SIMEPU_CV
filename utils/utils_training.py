import os
import pickle
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import albumentations
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
import sklearn.metrics as metrics

from utils.metrics import jaccard_coef, dice_coef
from utils.utils_data import SIMEPU_Dataset, SIMEPU_Dataset_MultiLabel, SIMEPU_Segmentation_Dataset


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
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 65, 100], gamma=0.1)
    elif plateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=6, factor=0.1, patience=12)
    else:
        return None


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_progress(epoch, progress_train_loss, progress_val_loss, progress_train_accuracy, progress_val_accuracy, model,
                  writer, output_dir, save_pickle=False):
    torch.save(model.state_dict(), output_dir + "/model_last.pt")

    writer.add_scalar('Loss/train', progress_train_loss[-1], epoch)
    writer.add_scalar('Loss/validation', progress_val_loss[-1], epoch)
    writer.add_scalar('Accuracy/train', progress_train_accuracy[-1], epoch)
    writer.add_scalar('Accuracy/validation', progress_val_accuracy[-1], epoch)

    if save_pickle:
        progress = {"train_loss": progress_train_loss, "val_loss": progress_val_loss,
                    "train_accuracy": progress_train_accuracy, "val_accuracy": progress_val_accuracy}

        with open(output_dir + 'progress.pickle', 'wb') as handle:
            pickle.dump(progress, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_overlays(img, mask, predicted_mask, save_path="", display=False):
    """
    """

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(18, 5))
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax5.axis('off')
    ax1.imshow(img)  # Imagen normal
    ax1.set_title("Original Image")
    ax2.imshow(mask, cmap="gray")  # Mascara original
    ax2.set_title("Original Mask")

    masked = np.ma.masked_where(mask == 0, mask)  # Overlay mascara original
    ax3.imshow(img, cmap="gray")
    ax3.imshow(masked, 'jet', interpolation='bilinear', alpha=0.25)
    ax3.set_title("Original Overlay")

    ax4.imshow(predicted_mask, cmap="gray")  # Mascara predecida
    ax4.set_title("Predicted Mask")

    masked = np.ma.masked_where(predicted_mask == 0, predicted_mask)  # Overlay mascara predecida
    ax5.imshow(img, cmap="gray")
    ax5.imshow(masked, 'jet', interpolation='bilinear', alpha=0.25)
    ax5.set_title("Predicted Overlay")

    if save_path != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    if display:
        plt.show()

    plt.close()


def reshape_masks(ndarray, to_shape):
    """
    Reshapes a center cropped (or padded) array back to its original shape.
    :param ndarray: (np.array) Mask Array to reshape
    :param to_shape: (tuple) Final desired shape
    :return: (np.array) Reshaped array to desired shape
    """
    h_in, w_in = ndarray.shape
    h_out, w_out = to_shape

    if h_in > h_out:  # center crop along h dimension
        h_offset = math.ceil((h_in - h_out) / 2)
        ndarray = ndarray[h_offset:(h_offset + h_out), :]
    else:  # zero pad along h dimension
        pad_h = (h_out - h_in)
        rem = pad_h % 2
        pad_dim_h = (math.ceil(pad_h / 2), math.ceil(pad_h / 2 + rem))
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, (0, 0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)

    if w_in > w_out:  # center crop along w dimension
        w_offset = math.ceil((w_in - w_out) / 2)
        ndarray = ndarray[:, w_offset:(w_offset + w_out)]
    else:  # zero pad along w dimension
        pad_w = (w_out - w_in)
        rem = pad_w % 2
        pad_dim_w = (math.ceil(pad_w / 2), math.ceil(pad_w / 2 + rem))
        npad = ((0, 0), pad_dim_w)
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)

    return ndarray  # reshaped


def train_step(train_loader, model, criterion, optimizer, segmentation_problem=False):
    model.train()
    if not segmentation_problem:
        train_loss, train_recall = 0, []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)

            targets = targets.squeeze().type_as(outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            outputs = (nn.Sigmoid()(outputs) > 0.5).float()
            train_recall.append(recall_score(targets.cpu().numpy(), outputs.cpu().numpy(), average="micro"))

        train_loss = (train_loss / (batch_idx + 1))
        train_recall = np.array(train_recall).mean()
        return train_loss, train_recall

    else:  # Segmentation problem / metrics
        train_loss, train_iou = 0, []
        for batch_idx, (inputs, _, masks, _, original_masks, _) in enumerate(train_loader):
            inputs, masks = inputs.cuda(), masks.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            for indx, single_pred in enumerate(outputs):
                original_mask = original_masks[indx]
                original_h, original_w = original_mask.shape
                resize_transform = albumentations.Resize(original_h, original_w)
                pred_mask = resize_transform(image=torch.sigmoid(single_pred).squeeze(0).data.cpu().numpy())["image"]
                binary_ground_truth = np.where(original_mask > 0.5, 1, 0).astype(np.int32)
                binary_pred_mask = np.where(pred_mask > 0.5, 1, 0).astype(np.int32)

                tmp_iou = jaccard_coef(binary_ground_truth, binary_pred_mask)
                train_iou.append(tmp_iou)

        train_loss = (train_loss / (batch_idx + 1))
        return train_loss, np.array(train_iou).mean()


def val_step(val_loader, model, criterion, segmentation_problem=False,
             masks_overlays=0, overlays_path="overlays", selected_class="", epoch=-1, lr=0):
    model.eval()
    if not segmentation_problem:
        val_loss, correct, total = 0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                # outputs = (nn.Sigmoid()(outputs) > 0.5).float()
                targets = targets.squeeze().type_as(outputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

                y_true.extend(targets.tolist())
                y_pred.extend(outputs.tolist())

            # avg = "micro"
            val_loss = (val_loss / (batch_idx + 1))

            y_true = np.stack(y_true)  # (batch, classes)
            y_pred = np.stack(y_pred)  # (batch, classes)

            # global_auc = metrics.roc_auc_score(y_true, y_pred)
            auc_per_class = []
            for c in range(y_true.shape[1]):
                auc_per_class.append(metrics.roc_auc_score(y_true[:, c], y_pred[:, c]))

            # val_precision_score = precision_score(y_true, y_pred, average=avg)
            # val_recall_score = recall_score(y_true, y_pred, average=avg)
            # val_f1_score = f1_score(y_true, y_pred, average=avg)
        # To generalize, as segmentation same number of outputs
        return val_loss, auc_per_class, None
        # return val_loss, val_recall_score, val_precision_score, val_recall_score, val_f1_score, None

    else:  # Segmentation problem
        val_loss, val_iou, val_dice, generated_masks, = 0, [], [], 0
        with torch.no_grad():
            for batch_idx, (inputs, _, masks, original_imgs, original_masks, inputs_names) in enumerate(val_loader):
                inputs, masks = inputs.cuda(), masks.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, masks)

                val_loss += loss.item()

                for indx, single_pred in enumerate(outputs):
                    original_mask = original_masks[indx]
                    original_h, original_w = original_mask.shape
                    resize_transform = albumentations.Resize(original_h, original_w)
                    pred_mask = resize_transform(image=torch.sigmoid(single_pred).squeeze(0).data.cpu().numpy())[
                        "image"]
                    binary_ground_truth = np.where(original_mask > 0.5, 1, 0).astype(np.int32)
                    binary_pred_mask = np.where(pred_mask > 0.5, 1, 0).astype(np.int32)

                    tmp_iou = jaccard_coef(binary_ground_truth, binary_pred_mask)
                    val_iou.append(tmp_iou)
                    tmp_dice = dice_coef(binary_ground_truth, binary_pred_mask)
                    val_dice.append(tmp_dice)

                    if generated_masks < masks_overlays:
                        save_overlays(
                            original_imgs[indx],
                            binary_ground_truth, binary_pred_mask,
                            os.path.join(
                                overlays_path, selected_class, f"{lr}", f"epoch{epoch}",
                                f"{inputs_names[indx].split('/')[-1]}"
                            )
                        )
                        generated_masks += 1

            val_loss = (val_loss / (batch_idx + 1))
        # To generalize, as classification same number of outputs
        return val_loss, np.array(val_iou).mean(), None, None, None, np.array(val_dice).mean()


def train_analysis(model, val_loader, output_dir, LABELS2TARGETS, TARGETS2LABELS, nb_classes):
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

    LABELS = []
    for i in range(len(LABELS2TARGETS)):
        print('Accuracy of {} : {:.2f}% '.format(TARGETS2LABELS[i], accuracy_per_class[i]))
        LABELS.append(TARGETS2LABELS[i])

    sns.set(style="whitegrid")
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax = sns.barplot(ax=ax, x=np.arange(len(LABELS2TARGETS)), y=accuracy_per_class)
    plt.xticks(np.arange(len(LABELS)), LABELS)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75)
    plt.xlabel("Clase")
    plt.ylabel("Accuracy (%)")

    axes = plt.gca()
    axes.set_ylim([0, 100])
    plt.yticks(np.arange(0, 101, 10.0))

    plt.savefig("{}/accuracy_per_class.png".format(output_dir), bbox_inches="tight")

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(val_loader):
            inputs, classes = inputs.cuda(), classes.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    # Great cmaps -> YlGnBu / GnBu
    ax = sns.heatmap(confusion_matrix.data.cpu().numpy() / np.array(class_total), cmap="YlGnBu", annot=True,
                     linewidths=.5)
    ax.set_yticklabels(ax.get_xticklabels(), rotation=0)
    plt.yticks(np.arange(len(LABELS)), LABELS)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.xticks(np.arange(len(LABELS)), LABELS)

    trans = mtrans.Affine2D().translate(30, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() + trans)

    trans = mtrans.Affine2D().translate(0, -25)
    for t in ax.get_yticklabels():
        t.set_transform(t.get_transform() + trans)

    plt.savefig("{}/confusion_matrix.png".format(output_dir), bbox_inches="tight")


def dataset_selector(train_aug, train_albumentation, val_aug, val_albumentation, args):
    if args.segmentation_problem and args.selected_class == "Grietas":

        train_dataset_longitudinales = SIMEPU_Segmentation_Dataset(
            data_partition='train', transform=train_aug, fold=args.fold, augmentation=train_albumentation,
            selected_class="Grietas longitudinales", data_mod=args.data_mod,
        )

        train_albumentation.append(albumentations.Rotate(limit=(90, 90), p=1))
        train_dataset_transversales = SIMEPU_Segmentation_Dataset(
            data_partition='train', transform=train_aug, fold=args.fold, augmentation=train_albumentation,
            selected_class="Grietas transversales", rotate=True, data_mod=args.data_mod,
        )

        train_dataset = torch.utils.data.ConcatDataset([train_dataset_longitudinales, train_dataset_transversales])

        num_classes = 1

        val_dataset_longitudinales = SIMEPU_Segmentation_Dataset(
            data_partition='validation', transform=val_aug, fold=args.fold, augmentation=val_albumentation,
            selected_class="Grietas longitudinales", data_mod=args.data_mod,
        )

        val_albumentation.append(albumentations.Rotate(limit=(90, 90), p=1))
        val_dataset_transversales = SIMEPU_Segmentation_Dataset(
            data_partition='validation', transform=val_aug, fold=args.fold, augmentation=val_albumentation,
            selected_class="Grietas transversales", rotate=True, data_mod=args.data_mod,
        )

        val_dataset = torch.utils.data.ConcatDataset([val_dataset_longitudinales, val_dataset_transversales])

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, pin_memory=True,
            shuffle=True, collate_fn=train_dataset_longitudinales.segmentation_collate
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, pin_memory=True,
            shuffle=False, collate_fn=val_dataset_longitudinales.segmentation_collate
        )
    elif args.segmentation_problem:

        train_dataset = SIMEPU_Segmentation_Dataset(
            data_partition='train', transform=train_aug, fold=args.fold,
            augmentation=train_albumentation, selected_class=args.selected_class
        )

        num_classes = train_dataset.num_classes

        val_dataset = SIMEPU_Segmentation_Dataset(
            data_partition='validation', transform=val_aug, fold=args.fold,
            augmentation=val_albumentation, selected_class=args.selected_class,
        )

        if not args.segmentation_problem:
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True,
            )
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)

        else:
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, pin_memory=True,
                shuffle=True, collate_fn=train_dataset.segmentation_collate
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, pin_memory=True,
                shuffle=False, collate_fn=val_dataset.segmentation_collate
            )

        return train_dataset, train_loader, val_dataset, val_loader, num_classes
    else:
        train_dataset = SIMEPU_Dataset_MultiLabel(
            data_partition='train', transform=train_aug, fold=args.fold,
            augmentation=train_albumentation, segmentation_problem=args.segmentation_problem,
            selected_class=args.selected_class
        )

        num_classes = train_dataset.num_classes

        val_dataset = SIMEPU_Dataset_MultiLabel(
            data_partition='validation', transform=val_aug, fold=args.fold,
            segmentation_problem=args.segmentation_problem,
            augmentation=val_albumentation, selected_class=args.selected_class,
        )

        if not args.segmentation_problem:
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, pin_memory=True,
                shuffle=True,
            )
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)

        else:
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, pin_memory=True,
                shuffle=True, collate_fn=train_dataset.segmentation_collate
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, pin_memory=True,
                shuffle=False, collate_fn=val_dataset.segmentation_collate
            )

    return train_dataset, train_loader, val_dataset, val_loader, num_classes
