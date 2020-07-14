#!/usr/bin/env python
# coding: utf-8

"""
```
python predict.py --img_sample samples/grieta1.jpg --download_models
python predict.py --img_sample samples/grieta1.jpg
```
"""

import sys
import os
from collections import OrderedDict
import argparse
import warnings
from skimage import io

import torch
import torchvision.models as models
from torchvision import transforms

warnings.filterwarnings('ignore')


def set_args():
    parser = argparse.ArgumentParser(description='SIMEPU Single Image prediction')
    parser.add_argument('--download_models', action='store_true', help='If used, download pretrained models')
    parser.add_argument("--binary_checkpoint", type=str, default="checkpoints/resnet34_binary.pt",
                        help="Checkpoint of model trained in damage vs. no damage case")
    parser.add_argument("--binary_threshold", type=float, default=0.8,
                        help="Binary to set samples as damage in the binary damages vs. no damages case")
    parser.add_argument("--damages_checkpoint", type=str, default="checkpoints/resnet34_damages.pt",
                        help="Checkpoint of model trained using only damages")
    parser.add_argument('--img_sample', type=str, required=True, help='Image to analyze/predict')
    arguments = parser.parse_args()
    return arguments


args = set_args()

print("\n-- SETTINGS --")
for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

if args.download_models:
    print("\nDownloading models...")
    os.system("chmod +x get_models.sh")
    os.system("bash get_models.sh")
    print("\nDone!")

if not os.path.isfile(args.binary_checkpoint) or not os.path.isfile(args.damages_checkpoint):
    assert False, "Unable to find model checkpoints. You can use pretrained models with --download_models"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_transforms = [
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]


def load_dataparallel_model(model, checkpoint):
    new_state_dict = OrderedDict()

    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model


num_classes = 1
binary_resnet34 = models.resnet34(pretrained=False)
binary_resnet34.fc = torch.nn.Linear(binary_resnet34.fc.in_features, num_classes)
binary_resnet34 = load_dataparallel_model(binary_resnet34, torch.load(args.binary_checkpoint))
binary_resnet34.to(DEVICE)
binary_resnet34.eval()

img = io.imread(args.img_sample)
img = transforms.Compose(img_transforms)(img)
batch = torch.unsqueeze(img, 0)

with torch.no_grad():
    y = torch.nn.Sigmoid()(binary_resnet34(batch.to(DEVICE)))[0]  # Only 1 sample, take it by index
    binary_out = (y > args.binary_threshold).int()

print("\n-> Probability of damage sample: {:.2f}%".format(y.item()*100))

if binary_out == 0:
    print("\n-- No damage sample --\n")
    sys.exit()

# Damage case
num_classes = 6
damages_resnet34 = models.resnet34(pretrained=False)
damages_resnet34.fc = torch.nn.Linear(damages_resnet34.fc.in_features, num_classes)
damages_resnet34 = load_dataparallel_model(damages_resnet34, torch.load(args.damages_checkpoint))
damages_resnet34.to(DEVICE)
damages_resnet34.eval()

with torch.no_grad():
    y = damages_resnet34(batch.to(DEVICE))[0]  # Only 1 sample, take it by index

TARGETS2LABELSDAMAGED = {
    0: 'Parcheo', 1: 'Grietas transversales', 2: 'Huecos', 3: 'Grietas longitudinales',
    4: 'Meteorización y desprendimiento', 5: 'Grietas en forma de piel de cocodrilo'
}

_, indices = torch.sort(y, descending=True)

print("Classified as: {}\n".format(TARGETS2LABELSDAMAGED[int(indices[0].item())]))
