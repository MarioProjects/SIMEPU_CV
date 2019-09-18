from time import gmtime, strftime

import albumentations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# ---- My utils ----
from models import *
from utils.train_arguments import *
from utils.utils_data import *
from utils.utils_training import *

if args.data_augmentation:
    train_aug = transforms.Compose([
        transforms.ToPILImage(), # because the input dtype is numpy.ndarray
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomCrop((args.crop_size, args.crop_size)),
        transforms.RandomHorizontalFlip(0.5), # because this method is used for PIL Image dtype
        transforms.ToTensor(), # because inpus dtype is PIL Image
    ])
else:
    train_aug = transforms.Compose([
        transforms.ToPILImage(),  # because the input dtype is numpy.ndarray
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomCrop((args.crop_size, args.crop_size)),
        transforms.RandomHorizontalFlip(0.5),  # because this method is used for PIL Image dtype
        transforms.ToTensor(),  # because inpus dtype is PIL Image
    ])

val_aug = transforms.Compose([
    transforms.ToPILImage(), # because the input dtype is numpy.ndarray
    transforms.Resize((args.img_size, args.img_size)),
    transforms.CenterCrop((args.crop_size, args.crop_size)),
    transforms.ToTensor(), # because inpus dtype is PIL Image
])

# data_partition='', data_augmentation=None, validation_size=0.15, seed=42
train_dataset = SIMEPU_Dataset(data_partition='train', transform=train_aug, validation_size=args.validation_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)

val_dataset = SIMEPU_Dataset(data_partition='validation', transform=val_aug, validation_size=args.validation_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)

# LABELS2TARGETS: Diccionario {"clase":target} para conocer el target que representa una clase dada
model = model_selector(args.model_name, num_classes=len(LABELS2TARGETS), pretrained=args.pretrained)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

progress_train_loss, progress_val_loss, = np.array([]), np.array([])
progress_train_accuracy, progress_val_accuracy = np.array([]), np.array([])
best_model = None

criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(args.optimizer, model, lr=args.learning_rate)
scheduler = get_scheduler(optimizer, args.steps_scheduler, args.plateau_scheduler)

print("\n--------------------------------------------")
for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

print("\n-------------- START TRAINING --------------")
for epoch in range(args.epochs):

    current_train_loss, current_train_accuracy = train_step(train_loader, model, criterion, optimizer)
    current_val_loss, current_val_accuracy = val_step(val_loader, model, criterion)

    progress_train_loss = np.append(progress_train_loss, current_train_loss)
    progress_val_loss = np.append(progress_val_loss, current_val_loss)
    progress_train_accuracy = np.append(progress_train_accuracy, current_train_accuracy)
    progress_val_accuracy = np.append(progress_val_accuracy, current_val_accuracy)

    save_progress(progress_train_loss, progress_val_loss, progress_train_accuracy, progress_val_accuracy, model, args.output_dir)

    if current_val_accuracy >= progress_val_accuracy.max():
        torch.save(model.state_dict(), args.output_dir + "/model_best_accuracy.pt")

    # Print training logs
    current_time = "[" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "]"
    print("{} Epoch: {}, LR: {:.8f}, Train Accuracy: {:.4f}, Val Accuracy: {:.4f}".format(
        current_time, epoch + 1, get_current_lr(optimizer), current_train_accuracy, current_val_accuracy
    ))

    if args.steps_scheduler:
        scheduler.step()
    elif args.plateau_scheduler:
        scheduler.step(current_val_accuracy)

print("\n------------------------------------------------")
print("Best Validation Accuracy {:.4f} at epoch {}".format(progress_val_accuracy.max(),
                                                           progress_val_accuracy.argmax() + 1))
print("------------------------------------------------\n")

print("---------------- Train Analysis ----------------")
train_analysis(model, val_loader, args.output_dir, LABELS2TARGETS, TARGETS2LABELS)