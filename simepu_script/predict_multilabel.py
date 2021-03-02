#!/usr/bin/env python
# coding: utf-8

"""
```
python predict_multilabel.py --img_sample samples/grieta1.jpg --download_models
python predict_multilabel.py --img_sample samples/grieta1.jpg
python predict_multilabel.py --img_sample samples/cocodrilo2.jpg --get_overlay
python predict_multilabel.py --img_sample samples/multiple1.jpg download_models --get_overlay
```
"""

import os
import argparse
import warnings
from skimage import io
import copy
import torchvision.models as models
from torchvision import transforms

from models import ExtraSmallUNet
from utils import *
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def set_args():
    parser = argparse.ArgumentParser(description='SIMEPU Single Image Multilabel prediction')
    parser.add_argument('--download_models', action='store_true', help='If used, download pretrained models')
    parser.add_argument("--multilabel_checkpoint", type=str, default="checkpoints/resnet34_multilabel.pt",
                        help="Checkpoint of model trained in damage vs. no damage case")
    parser.add_argument("--multilabel_threshold", type=float, default=0.83,
                        help="Threshold for multilabel classification")
    parser.add_argument("--huecos_segmentation_checkpoint", type=str, default="checkpoints/huecos_segmentation.pt",
                        help="Checkpoint of model trained using segmentation of huecos")
    parser.add_argument("--parcheo_segmentation_checkpoint", type=str, default="checkpoints/parcheo_segmentation.pt",
                        help="Checkpoint of model trained using segmentation of parcheo")
    parser.add_argument("--transversales_segmentation_checkpoint", type=str,
                        default="checkpoints/transversales_segmentation.pt",
                        help="Checkpoint of model trained using segmentation of transversales")
    parser.add_argument("--longitudinales_segmentation_checkpoint", type=str,
                        default="checkpoints/longitudinales_segmentation.pt",
                        help="Checkpoint of model trained using segmentation of longitudinales")
    parser.add_argument('--get_overlay', action='store_true', help='If used, return overlay in case segmentable damage')
    parser.add_argument('--img_sample', type=str, required=True, help='Image to analyze/predict')
    arguments = parser.parse_args()
    return arguments


args = set_args()

print("\n-- SETTINGS --")
for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

if args.download_models:
    print("\nDownloading models...")
    os.system("chmod +x get_multilabel_models.sh")
    os.system("bash get_multilabel_models.sh")
    print("\nDone!")

if not os.path.isfile(args.multilabel_checkpoint):
    assert False, "Unable to find model checkpoints. You can use pretrained models with --download_models"


CLASSES = [
    "Alcantarillado", "Marca vial", "Hueco", "Parcheo", "Grietas longitudinales",
    "Grietas en forma de piel de cocodrilo", "Grietas transversales",
    "Meteorización y desprendimiento"
]


##########################################################################################
# -------------------------- MULTILABEL CLASSIFICATION CASE ------------------------------
##########################################################################################

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_transforms = [
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

num_classes = len(CLASSES)
multilabel_resnet34 = models.resnet34(pretrained=False)
multilabel_resnet34.fc = torch.nn.Linear(multilabel_resnet34.fc.in_features, num_classes)
multilabel_resnet34 = load_dataparallel_model(multilabel_resnet34, torch.load(args.multilabel_checkpoint))
multilabel_resnet34.to(DEVICE)
multilabel_resnet34.eval()


img = io.imread(args.img_sample)
img_t = transforms.Compose(img_transforms)(copy.deepcopy(img))
batch = torch.unsqueeze(img_t, 0)

with torch.no_grad():
    y = torch.nn.Sigmoid()(multilabel_resnet34(batch.to(DEVICE)))[0]  # Only 1 sample, take it by index
    multilabel_out = (y > args.multilabel_threshold).int()

str_res = ""
for pred_index, pred in enumerate(multilabel_out):
    str_res = f"{str_res}\n{CLASSES[pred_index]}" if pred else str_res

print(f"Clasificado como: {str_res}")


##########################################################################################
# ------------------------------- SEGMENTATION CASE --------------------------------------
##########################################################################################

img_size = 512
val_albumentation = [
    albumentations.Resize(img_size, img_size),
]

img_t, _ = apply_augmentations(copy.deepcopy(img), albumentations.Compose(val_albumentation), None, None)
img_t = apply_normalization(img, "reescale")
img_t = torch.from_numpy(img_t.transpose(2, 0, 1)).float()  # Transpose == Channels first
batch = torch.unsqueeze(img_t, 0)

num_classes = 1
model = ExtraSmallUNet(n_channels=3, n_classes=num_classes)

final_mask = np.zeros(img_t.shape[1:])

overlay_mask_info = ""
# Parcheo
parcheo_index = CLASSES.index("Parcheo")
if multilabel_out[parcheo_index]:
    mask = load_predict_segmentation(model, args.parcheo_segmentation_checkpoint, img, batch, DEVICE)
    pixeles_parcheo = mask.sum()
    porcentaje_parcheo = (pixeles_parcheo / (mask.shape[0] * mask.shape[1])) * 100
    overlay_mask_info = "- {:.2f}%".format(porcentaje_parcheo)
    print(f"El parcheo esta compuesto por {pixeles_parcheo} pixeles, el {porcentaje_parcheo:.2f}% de la imagen")

    final_mask = np.logical_or(final_mask, mask) * 1

# Grietas transversales
transversales_index = CLASSES.index("Grietas transversales")
if multilabel_out[transversales_index]:
    mask = load_predict_segmentation(model, args.transversales_segmentation_checkpoint, img, batch, DEVICE)

    columns_indices = np.where(np.any(mask, axis=0))[0]
    rows_indices = np.where(np.any(mask, axis=1))[0]

    first_column_index, last_column_index = columns_indices[0], columns_indices[-1]
    first_row_index, last_row_index = rows_indices[0], rows_indices[-1]

    anchura = abs(first_row_index - last_row_index)
    longitud = abs(first_column_index - last_column_index)
    overlay_mask_info = f" - Longitud {longitud} y Anchura {anchura}"
    print(f"La grieta tiene una longitud de {longitud} pixeles y abarca {anchura} pixeles de ancho")

    final_mask = np.logical_or(final_mask, mask) * 1

# Hueco
hueco_index = CLASSES.index("Hueco")
if multilabel_out[hueco_index]:

    mask = load_predict_segmentation(model, args.huecos_segmentation_checkpoint, img, batch, DEVICE)
    pixeles_hueco = mask.sum()
    porcentaje_hueco = (pixeles_hueco / (mask.shape[0] * mask.shape[1])) * 100
    overlay_mask_info = "- {:.2f}%".format(porcentaje_hueco)
    print(f"El hueco esta compuesto por {pixeles_hueco} pixeles, el {porcentaje_hueco:.2f}% de la imagen")

    final_mask = np.logical_or(final_mask, mask) * 1


# Grietas longitudinales
longitudinales_index = CLASSES.index("Grietas longitudinales")
if multilabel_out[longitudinales_index]:

    mask = load_predict_segmentation(model, args.longitudinales_segmentation_checkpoint, img, batch, DEVICE)

    columns_indices = np.where(np.any(mask, axis=0))[0]
    rows_indices = np.where(np.any(mask, axis=1))[0]

    first_column_index, last_column_index = columns_indices[0], columns_indices[-1]
    first_row_index, last_row_index = rows_indices[0], rows_indices[-1]

    longitud = abs(first_row_index - last_row_index)
    anchura = abs(first_column_index - last_column_index)
    overlay_mask_info = f" - Longitud {longitud} y Anchura {anchura}"
    print(f"La grieta tiene una longitud de {longitud} pixeles y abarca {anchura} pixeles de ancho")

    final_mask = np.logical_or(final_mask, mask) * 1

# Grietas en forma de piel de cocodrilo
cocodrilo_index = CLASSES.index("Grietas en forma de piel de cocodrilo")
if multilabel_out[cocodrilo_index]:
    mask_l = mask = load_predict_segmentation(model, args.longitudinales_segmentation_checkpoint, img, batch, DEVICE)
    mask_t = mask = load_predict_segmentation(model, args.transversales_segmentation_checkpoint, img, batch, DEVICE)
    mask = np.logical_or(mask_l, mask_t) * 1

    pixeles_cocodrilo = mask.sum()
    porcentaje_cocodrilo = (pixeles_cocodrilo / (mask.shape[0] * mask.shape[1])) * 100
    overlay_mask_info = "- {:.2f}%".format(porcentaje_cocodrilo)
    print(f"El cocodrilo esta compuesto por {pixeles_cocodrilo} pixeles, el {porcentaje_cocodrilo:.2f}% de la imagen")

    final_mask = np.logical_or(final_mask, mask) * 1

if args.get_overlay:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax1.imshow(img)  # Imagen normal
    ax1.set_title("Original Image")

    ax2.imshow(final_mask, cmap="gray")  # Mascara predicha
    ax2.set_title("Predicted Mask")

    masked = np.ma.masked_where(final_mask == 0, final_mask)  # Overlay mascara predicha
    ax3.imshow(img, cmap="gray")
    ax3.imshow(masked, 'jet', interpolation='bilinear', alpha=0.35)
    ax3.set_title(f"Predicted Overlay{overlay_mask_info}")
    os.makedirs("overlays", exist_ok=True)
    plt.savefig(
        f"overlays/{os.path.splitext(os.path.basename(args.img_sample))[0]}_overlay_segmentacion_multilabel.png"
    )
