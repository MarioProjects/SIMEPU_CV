#!/usr/bin/env python
# coding: utf-8
from time import gmtime, strftime

import pretty_errors

# ---- My utils ----
from models import *
from utils.train_arguments import *
from utils.utils_training import *
from utils.data_augmentation import get_augmentations

pretty_errors.mono()

train_aug, val_aug, train_albumentation, val_albumentation = get_augmentations(
    args.pretrained, args.img_size, args.segmentation_problem,
    args.randaug_n, args.randaug_m, args.cutout_size
)

train_dataset, train_loader, val_dataset, val_loader, num_classes = dataset_selector(
    train_aug, train_albumentation, val_aug, val_albumentation, args
)

print("There are {} classes!".format(num_classes))
print("[Train fold] {} samples".format(len(train_dataset)))
print("[Validation fold] {} samples".format(len(val_dataset)))

model = model_selector(
    args.model_name, num_classes=num_classes, pretrained=args.pretrained, scale_factor=args.unet_scale_factor
)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

if args.checkpoint != "":
    model.load_state_dict(torch.load(args.checkpoint))
    print(f"Model checkpoint loaded correctly from: {args.checkpoint}")

max_metric, max_metric_epoch, max_metric_str_logline = 0, 0, ""
best_model = None

if args.binary_problem or args.segmentation_problem:
    criterion = nn.BCEWithLogitsLoss()
elif args.multilabel_problem:
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

print("\n-------------- START TRAINING --------------")
for epoch in range(args.epochs):

    current_train_loss, current_train_metric = train_step(
        train_loader, model, criterion, optimizer,
        segmentation_problem=args.segmentation_problem
    )

    # current_val_metric, val_precision_score, val_recall_score, val_f1_score
    current_val_loss, auc_per_class, val_dice = val_step(
        val_loader, model, criterion, args.segmentation_problem,
        selected_class=args.selected_class, masks_overlays=args.masks_overlays, epoch=(epoch+1), lr=args.learning_rate
    )

    # -- Print training logs --
    current_time = "[" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "]"
    str_epoch = epoch + 1
    str_logline = f"{current_time} Epoch: {str_epoch}"
    if args.binary_problem or args.multilabel_problem:

        str_logline = "{} Epoch: {}, LR: {:.8f}, Train Recall: {:.4f}".format(
            current_time, str_epoch, get_current_lr(optimizer), current_train_metric
        )

        # val_metrics = ", Val Recall: {:.4f}, Val Precision: {:.4f}, Val F1: {:.4f}".format(
        #     val_recall_score, val_precision_score, val_f1_score
        # )

        current_val_metric = np.array(auc_per_class).mean()
        val_metrics = " ".join([f"{a}: {b:.4f}" for a, b in [x for x in zip(train_dataset.classes, auc_per_class)]])

        str_logline += f", {val_metrics}"

    elif args.segmentation_problem:
        current_val_metric = val_dice
        str_logline = "{} Epoch: {}, LR: {:.8f}, Train IOU: {:.4f}, Val IOU: {:.4f}, Val DICE: {:.4f}".format(
            current_time, str_epoch, get_current_lr(optimizer), current_train_metric, current_val_metric, val_dice
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

    torch.save(model.state_dict(), args.output_dir + "/model_last.pt")

print("\n------------------------------------------------")
print(f"Best Validation:\n\t{max_metric_str_logline}")
print("------------------------------------------------\n")

