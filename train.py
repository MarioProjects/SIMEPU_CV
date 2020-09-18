#!/usr/bin/env python
# coding: utf-8
from time import gmtime, strftime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---- My utils ----
from models import *
from utils.train_arguments import *
from utils.utils_data import *
from utils.utils_training import *
from utils.data_augmentation import get_augmentations

from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss


train_aug, val_aug, train_albumentation, val_albumentation = get_augmentations(
    args.data_augmentation, args.pretrained, args.img_size, args.crop_size, args.segmentation_problem
)

train_dataset = SIMEPU_Dataset(data_partition='train', transform=train_aug,
                               validation_size=args.validation_size, binary_problem=args.binary_problem,
                               damaged_problem=args.damaged_problem, segmentation_problem=args.segmentation_problem,
                               augmentation=train_albumentation, selected_class=args.selected_class)

num_classes = train_dataset.num_classes

if args.cutmix:
    train_dataset = CutMix(
        train_dataset, num_class=num_classes if not args.binary_problem else 2, beta=1.0, prob=0.65, num_mix=1
    )

val_dataset = SIMEPU_Dataset(data_partition='validation', transform=val_aug,
                             validation_size=args.validation_size, binary_problem=args.binary_problem,
                             damaged_problem=args.damaged_problem, segmentation_problem=args.segmentation_problem,
                             augmentation=val_albumentation, selected_class=args.selected_class)

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



print("Hay {} clases!".format(num_classes))
print("[Train] {} muestras".format(len(train_dataset)))
print("[Validacion] {} muestras".format(len(val_dataset)))

model = model_selector(args.model_name, num_classes=num_classes, pretrained=args.pretrained)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

progress_train_loss, progress_val_loss, = np.array([]), np.array([])
progress_train_metric, progress_val_metric = np.array([]), np.array([])
best_model = None

# if args.binary_problem and args.cutmix:
#     assert False, "Not implemented binary problem with cutmix!"

if args.cutmix:
    criterion = CutMixCrossEntropyLoss(True)
elif args.binary_problem or args.segmentation_problem:
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

if args.segmentation_problem:
    metric_str = "IOU"
else:
    metric_str = "Accuracy"

print("\n-------------- START TRAINING --------------")
for epoch in range(args.epochs):

    current_train_loss, current_train_metric = train_step(
        train_loader, model, criterion, optimizer,
        binary_problem=args.binary_problem, segmentation_problem=args.segmentation_problem, cutmix=args.cutmix
    )
    current_val_loss, current_val_metric = val_step(
        val_loader, model, criterion, args.binary_problem, args.segmentation_problem,
        selected_class=args.selected_class, masks_overlays=args.masks_overlays, epoch=(epoch+1), lr=args.learning_rate
    )

    progress_train_loss = np.append(progress_train_loss, current_train_loss)
    progress_val_loss = np.append(progress_val_loss, current_val_loss)
    progress_train_metric = np.append(progress_train_metric, current_train_metric)
    progress_val_metric = np.append(progress_val_metric, current_val_metric)

    save_progress(
        epoch, progress_train_loss, progress_val_loss,
        progress_train_metric, progress_val_metric, model, writer, args.output_dir
    )

    if current_val_metric >= progress_val_metric.max():
        torch.save(model.state_dict(), args.output_dir + "/model_best_metric.pt")

    # Print training logs
    current_time = "[" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "]"
    print("{} Epoch: {}, LR: {:.8f}, Train {}: {:.4f}, Val {}: {:.4f}".format(
        current_time, epoch + 1, get_current_lr(optimizer),
        metric_str, current_train_metric, metric_str, current_val_metric
    ))

    if args.steps_scheduler:
        scheduler.step()
    elif args.plateau_scheduler:
        scheduler.step(current_val_metric)

print("\n------------------------------------------------")
print("Best Validation {} {:.4f} at epoch {}".format(
    metric_str, progress_val_metric.max(), progress_val_metric.argmax() + 1)
)
print("------------------------------------------------\n")

if not args.binary_problem and not args.segmentation_problem:
    print("---------------- Train Analysis ----------------")
    if args.damaged_problem:
        train_analysis(model, val_loader, args.output_dir, LABELS2TARGETSDAMAGED, TARGETS2LABELSDAMAGED, 6)
    else:
        train_analysis(model, val_loader, args.output_dir, LABELS2TARGETS, TARGETS2LABELS, 9)
