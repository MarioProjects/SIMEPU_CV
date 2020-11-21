#!/usr/bin/env python
# coding: utf-8
from time import gmtime, strftime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# ---- My utils ----
from models import *
from utils.train_arguments import *
from utils.utils_data import *
from utils.utils_training import *
from utils.data_augmentation import get_augmentations

train_aug, val_aug, train_albumentation, val_albumentation = get_augmentations(
    args.data_augmentation, args.pretrained, args.img_size, args.crop_size, args.segmentation_problem
)

train_dataset, train_loader, val_dataset, val_loader, num_classes = dataset_selector(
    train_aug, train_albumentation, val_aug, val_albumentation, args
)

print("There are {} classes!".format(num_classes))
print("[Train fold] {} samples".format(len(train_dataset)))
print("[Validation fold] {} samples".format(len(val_dataset)))

model = model_selector(args.model_name, num_classes=num_classes, pretrained=args.pretrained)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

max_metric, max_metric_epoch, max_metric_str_logline = 0, 0, ""
best_model = None

if args.binary_problem or args.segmentation_problem:
    criterion = nn.BCEWithLogitsLoss()
elif args.weighted_loss:
    print("Loaded Class weights!")
    with open("utils/class_weights_divide.pkl", "rb") as fp:  # Unpickling
        weights = pickle.load(fp)
    print("Weights: {}".format(weights))
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device='cuda'))
else:
    criterion = nn.CrossEntropyLoss()

optimizer = get_optimizer(args.optimizer, model, lr=args.learning_rate)
scheduler = get_scheduler(optimizer, args.steps_scheduler, args.plateau_scheduler)

print("\n--------------------------------------------")
for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))
writer = SummaryWriter(log_dir='results/logs/{}'.format(args.output_dir[args.output_dir.find("results/") + 8:-1]))

print("\n-------------- START TRAINING --------------")
for epoch in range(args.epochs):

    current_train_loss, current_train_metric = train_step(
        train_loader, model, criterion, optimizer,
        binary_problem=args.binary_problem, segmentation_problem=args.segmentation_problem
    )
    current_val_loss, current_val_metric, val_precision_score, val_recall_score, val_f1_score, val_balanced_accuracy_score, val_dice = val_step(
        val_loader, model, criterion, args.binary_problem, args.segmentation_problem,
        selected_class=args.selected_class, masks_overlays=args.masks_overlays, epoch=(epoch+1), lr=args.learning_rate
    )

    # -- Print training logs --
    if args.binary_problem:
        current_time = "[" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "]"
        str_logline = "{} Epoch: {}, LR: {:.8f}, Train Accuracy: {:.4f}, Val Accuracy: {:.4f}, Val Precision: {:.4f}, Val Recall: {:.4f}, Val F1: {:.4f}".format(
            current_time, epoch + 1, get_current_lr(optimizer),
            current_train_metric, current_val_metric, val_precision_score, val_recall_score, val_f1_score
        )

    elif args.segmentation_problem:
        current_time = "[" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "]"
        str_logline = "{} Epoch: {}, LR: {:.8f}, Train IOU: {:.4f}, Val IOU: {:.4f}, Val DICE: {:.4f}".format(
            current_time, epoch + 1, get_current_lr(optimizer),
            current_train_metric, current_val_metric, val_dice
        )
    else:  # Damage classification case
        current_time = "[" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "]"
        str_logline = "{} Epoch: {}, LR: {:.8f}, Train Accuracy: {:.4f}, Val Accuracy: {:.4f}, Val Balanced Accuracy: {:.4f}, Val Precision: {:.4f}, Val Recall: {:.4f}, Val F1: {:.4f}".format(
            current_time, epoch + 1, get_current_lr(optimizer),
            current_train_metric, current_val_metric, val_balanced_accuracy_score, val_precision_score, val_recall_score, val_f1_score
        )

    print(str_logline)
    if current_val_metric >= max_metric:
        torch.save(model.state_dict(), args.output_dir + "/model_best_metric.pt")
        max_metric = current_val_metric
        max_metric_epoch = epoch + 1
        max_metric_str_logline = str_logline

    if args.steps_scheduler:
        scheduler.step()
    elif args.plateau_scheduler:
        scheduler.step(current_val_metric)

print("\n------------------------------------------------")
print(f"Best Validation:\n\t{max_metric_str_logline}")
print("------------------------------------------------\n")

if not args.binary_problem and not args.segmentation_problem:
    print("---------------- Train Analysis ----------------")
    if args.damaged_problem:
        train_analysis(model, val_loader, args.output_dir, LABELS2TARGETSDAMAGED, TARGETS2LABELSDAMAGED, 6)
    else:
        train_analysis(model, val_loader, args.output_dir, LABELS2TARGETS, TARGETS2LABELS, 9)
