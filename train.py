#!/usr/bin/env python
# coding: utf-8
from time import gmtime, strftime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# ---- My utils ----
from models import *
from utils.train_arguments import *
from utils.utils_data import *
from utils.utils_training import *

if args.data_augmentation:
    train_aug = [
        transforms.ToPILImage(),  # because the input dtype is numpy.ndarray
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomCrop((args.crop_size, args.crop_size)),
        transforms.RandomHorizontalFlip(0.5),  # because this method is used for PIL Image dtype
        transforms.RandomVerticalFlip(0.5),  # because this method is used for PIL Image dtype
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),  # because inpus dtype is PIL Image
    ]
else:
    train_aug = [
        transforms.ToPILImage(),  # because the input dtype is numpy.ndarray
        transforms.Resize((args.img_size, args.img_size)),
        transforms.CenterCrop((args.crop_size, args.crop_size)),
        transforms.ToTensor(),  # because inpus dtype is PIL Image
    ]

val_aug = [
    transforms.ToPILImage(),  # because the input dtype is numpy.ndarray
    transforms.Resize((args.img_size, args.img_size)),
    transforms.CenterCrop((args.crop_size, args.crop_size)),
    transforms.ToTensor(),  # because inpus dtype is PIL Image
]

if args.pretrained:
    train_aug.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    val_aug.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

# data_partition='', data_augmentation=None, validation_size=0.15, seed=42
train_dataset = SIMEPU_Dataset(data_partition='train', transform=transforms.Compose(train_aug),
                               validation_size=args.validation_size, binary_problem=args.binary_problem,
                               damaged_problem=args.damaged_problem)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)

val_dataset = SIMEPU_Dataset(data_partition='validation', transform=transforms.Compose(val_aug),
                             validation_size=args.validation_size, binary_problem=args.binary_problem,
                             damaged_problem=args.damaged_problem)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)

print("[Train] {} muestras".format(len(train_dataset)))
print("[Validacion] {} muestras".format(len(val_dataset)))

print("Hay {} clases!".format(train_dataset.num_classes))
model = model_selector(args.model_name, num_classes=train_dataset.num_classes, pretrained=args.pretrained)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

progress_train_loss, progress_val_loss, = np.array([]), np.array([])
progress_train_accuracy, progress_val_accuracy = np.array([]), np.array([])
best_model = None

if args.binary_problem:
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

    current_train_loss, current_train_accuracy = train_step(train_loader, model, criterion, optimizer, binary_problem=args.binary_problem)
    current_val_loss, current_val_accuracy = val_step(val_loader, model, criterion, args.binary_problem)

    progress_train_loss = np.append(progress_train_loss, current_train_loss)
    progress_val_loss = np.append(progress_val_loss, current_val_loss)
    progress_train_accuracy = np.append(progress_train_accuracy, current_train_accuracy)
    progress_val_accuracy = np.append(progress_val_accuracy, current_val_accuracy)

    save_progress(epoch, progress_train_loss, progress_val_loss, progress_train_accuracy, progress_val_accuracy, model, writer, args.output_dir)

    if current_val_accuracy >= progress_val_accuracy.max():
        torch.save(model.state_dict(), args.output_dir + "/model_best_accuracy.pt")

    # Print training logs
    current_time = "[" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "]"
    print("{} Epoch: {}, LR: {:.8f}, Train Accuracy: {:.4f}%, Val Accuracy: {:.4f}%".format(
        current_time, epoch + 1, get_current_lr(optimizer), current_train_accuracy, current_val_accuracy
    ))

    if args.steps_scheduler:
        scheduler.step()
    elif args.plateau_scheduler:
        scheduler.step(current_val_accuracy)

print("\n------------------------------------------------")
print("Best Validation Accuracy {:.4f} at epoch {}".format(progress_val_accuracy.max(), progress_val_accuracy.argmax() + 1))
print("------------------------------------------------\n")

if not args.binary_problem:
    print("---------------- Train Analysis ----------------")
    if args.damaged_problem:
        train_analysis(model, val_loader, args.output_dir, LABELS2TARGETSDAMAGED, TARGETS2LABELSDAMAGED, 6)
    else:
        train_analysis(model, val_loader, args.output_dir, LABELS2TARGETS, TARGETS2LABELS, 9)
