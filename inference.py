#!/usr/bin/env python
# coding: utf-8

import time

from torchvision import transforms
from PIL import Image

import ttach as tta

from toma import toma

# ---- My utils ----
from simepu_script.utils import *
from simepu_script.models import *
import torchvision.models as models
from torch.utils.data import DataLoader

DATA_DIR = '/home/maparla/SIMEPU_JUN_2021/all/images'
batch_size_multilabel = 512
batch_size_severidades = 32

muestras_procesar = 1000  # Para procesar todas las muestras de DATA_DIR dejar a -1

model_multilabel_path = os.path.join("checkpoints", "resnet34_multilabel.pt")
model_parcheo_path = os.path.join("checkpoints", "parcheo_segmentation.pt")
model_hueco_path = os.path.join("checkpoints", "huecos_segmentation.pt")
model_transversales_path = os.path.join("checkpoints", "transversales_segmentation.pt")
model_longitudinales_path = os.path.join("checkpoints", "longitudinales_segmentation.pt")

# No tocar lo siguiente :)
img_size_multilabel = 224
img_size_severidades = 512
area_img = 512 * 512
pretrained = True

CLASSES = [
    "Alcantarillado", "Marca vial", "Hueco", "Parcheo", "Grietas longitudinales",
    "Grietas en forma de piel de cocodrilo", "Grietas transversales",
    "Meteorización y desprendimiento"
]

val_aug_multilabel = transforms.Compose([
    transforms.ToPILImage(),  # because the input dtype is numpy.ndarray
    transforms.Resize(img_size_multilabel, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_albumentation_severidades = [
    albumentations.Resize(img_size_severidades, img_size_severidades),
]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE == "cpu":
    assert False, "CPU MODE NOT AVAILABLE!"


def get_severidad_grieta(mask):
    columns_indices = np.where(np.any(mask, axis=0))[0]
    rows_indices = np.where(np.any(mask, axis=1))[0]

    if columns_indices.size and rows_indices.size:
        first_row_index, last_row_index = rows_indices[0], rows_indices[-1]

        longitud = abs(first_row_index - last_row_index)
        longitud_real = (longitud / 512) * 1

        anchos = []
        for co in rows_indices:
            indice_anchura = co
            anchura_real = np.sum(mask[int(indice_anchura), :])
            anchos.append((anchura_real / 512) * 1)

        anchura_real = np.array(anchos).mean()

        return {"longitud_real": longitud_real, "anchura_real": anchura_real}

    return {"longitud_real": 0, "anchura_real": 0}


def load_dataparallel_model(model, checkpoint):
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model


# ## Multilabel

# #### Load Data
inference_dataset = SIMEPU_Dataset_MultiLabel(
    from_folder=DATA_DIR, transform=val_aug_multilabel, fold=-1, get_path=True
)

# #### Load model
multilabel_resnet34 = models.resnet34(pretrained=False)
multilabel_resnet34.fc = torch.nn.Linear(multilabel_resnet34.fc.in_features, len(CLASSES))
multilabel_resnet34 = load_dataparallel_model(
    multilabel_resnet34, torch.load(model_multilabel_path)
)
multilabel_resnet34 = torch.nn.DataParallel(multilabel_resnet34, device_ids=range(torch.cuda.device_count()))
multilabel_resnet34.to(DEVICE)
multilabel_resnet34.eval()

multilabel_resnet34 = tta.ClassificationTTAWrapper(multilabel_resnet34, tta.Compose([
    tta.HorizontalFlip(), tta.VerticalFlip(), tta.Scale([0.85, 1.15]),
]), merge_mode="mean")

# #### Inference
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_th = torch.tensor([0.0005, 0.2, 0.005, 0.02, 0.005, 0.005, 0.008, 0.005]).to(DEVICE)

total_preds, total_paths = [], []


@toma.batch(initial_batchsize=batch_size_multilabel)
def run_multilabel(batch_size):
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    with torch.no_grad():
        for batch_idx, (inputs, _, paths) in enumerate(inference_loader):
            inputs = inputs.cuda()
            outputs = multilabel_resnet34(inputs)

            y = torch.nn.Sigmoid()(outputs)
            multilabel_out = (y > class_th).int()

            total_paths.extend(paths)
            total_preds.extend(multilabel_out.tolist())

            if muestras_procesar != -1 and muestras_procesar < (batch_idx + 1) * batch_size_multilabel:
                break


print("-- START MULTILABEL PREDICTION --")
start_total_time = time.time()
start_time = time.time()
run_multilabel()

multilabel_end = time.time()
preds_info = {"Image": [p.split("/")[-1] for p in total_paths]}
preds_info.update({clase: np.array(total_preds)[:, index] for index, clase in enumerate(CLASSES)})

labels_preds = pd.DataFrame(preds_info)

labels_preds["area_severidad_meteorizacion"] = 0
labels_preds["area_daño_meteorizacion"] = 0

labels_preds.loc[labels_preds["Meteorización y desprendimiento"] == 1, "area_severidad_meteorizacion"] = 1
labels_preds.loc[labels_preds["Meteorización y desprendimiento"] == 1, "area_daño_meteorizacion"] = 1

labels_preds["area_severidad_parcheo"] = 0
labels_preds["area_daño_parcheo"] = 0

labels_preds["area_severidad_hueco"] = 0
labels_preds["area_daño_hueco"] = 0

labels_preds["area_severidad_cocodrilo"] = 0
labels_preds["area_daño_cocodrilo"] = 0

labels_preds["area_severidad_transversales"] = 0
labels_preds["area_daño_transversales"] = 0

labels_preds["area_severidad_longitudinales"] = 0
labels_preds["area_daño_longitudinales"] = 0

print(f"Han sido clasificadas {len(labels_preds)} muestras")
print("--- INFERENCIA MULTIETIQUETA %s seconds ---" % (multilabel_end - start_time))

# labels_preds.to_csv("multilabel_inference.csv", index=None)
# sys.exit()

# ## Huecos

# #### Load Data
huecos_cases = labels_preds.loc[labels_preds["Hueco"] == 1]
print(f"\nExisten {len(huecos_cases)} muestras con huecos")
huecos_dataset = SIMEPU_Dataset_MultiLabel(
    from_df=huecos_cases, from_folder=DATA_DIR,
    augmentation=val_albumentation_severidades, fold=-1, get_path=True
)

# #### Load model
huecos_model = ExtraSmallUNet(n_channels=3, n_classes=1)
huecos_model = torch.nn.DataParallel(huecos_model, device_ids=range(torch.cuda.device_count()))
huecos_model.load_state_dict(torch.load(model_hueco_path))
huecos_model.to(DEVICE)
huecos_model.eval()

# #### Inference


total_paths, total_areas = [], []
start_time = time.time()


@toma.batch(initial_batchsize=batch_size_severidades)
def run_huecos(batch_size):
    huecos_loader = DataLoader(huecos_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    with torch.no_grad():
        for batch_idx, (inputs, _, paths) in enumerate(huecos_loader):
            inputs = inputs.cuda()
            outputs = huecos_model(inputs)

            y = torch.nn.Sigmoid()(outputs)
            masks = (y > 0.5).int()

            total_paths.extend(paths)
            total_areas.extend([(mask.sum().item() / area_img) for mask in masks])


run_huecos()

for path, area in zip(total_paths, total_areas):
    labels_preds.loc[labels_preds["Image"] == path.split("/")[-1], "area_severidad_hueco"] = area
    labels_preds.loc[labels_preds["Image"] == path.split("/")[-1], "area_daño_hueco"] = area

print("--- INFERENCIA HUECOS %s seconds ---" % (time.time() - start_time))

# ## Parcheo

# #### Load Data
parcheos_cases = labels_preds.loc[labels_preds["Parcheo"] == 1]
print(f"\nExisten {len(parcheos_cases)} muestras con parcheos")
parcheos_dataset = SIMEPU_Dataset_MultiLabel(
    from_df=parcheos_cases, from_folder=DATA_DIR,
    augmentation=val_albumentation_severidades, fold=-1, get_path=True
)

# #### Load model
parcheos_model = ExtraSmallUNet(n_channels=3, n_classes=1)
parcheos_model = torch.nn.DataParallel(parcheos_model, device_ids=range(torch.cuda.device_count()))
parcheos_model.load_state_dict(torch.load(model_parcheo_path))
parcheos_model.to(DEVICE)
parcheos_model.eval()

# #### Inference

total_paths, total_areas = [], []
start_time = time.time()


@toma.batch(initial_batchsize=batch_size_severidades)
def run_parcheos(batch_size):
    parcheos_loader = DataLoader(parcheos_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    with torch.no_grad():
        for batch_idx, (inputs, _, paths) in enumerate(parcheos_loader):
            inputs = inputs.cuda()
            outputs = parcheos_model(inputs)

            y = torch.nn.Sigmoid()(outputs)
            masks = (y > 0.5).int()

            total_paths.extend(paths)
            total_areas.extend([(mask.sum().item() / area_img) for mask in masks])


run_parcheos()

for path, area in zip(total_paths, total_areas):
    labels_preds.loc[labels_preds["Image"] == path.split("/")[-1], "area_severidad_parcheo"] = area
    labels_preds.loc[labels_preds["Image"] == path.split("/")[-1], "area_daño_parcheo"] = area

print("--- INFERENCIA PARCHEO %s seconds ---" % (time.time() - start_time))

# ## Longitudinales

# #### Load Data
longitudinales_cases = labels_preds.loc[labels_preds["Grietas longitudinales"] == 1]
print(f"\nExisten {len(longitudinales_cases)} muestras con grietas longitudinales")
longitudinales_dataset = SIMEPU_Dataset_MultiLabel(
    from_df=longitudinales_cases, from_folder=DATA_DIR,
    augmentation=val_albumentation_severidades, fold=-1, get_path=True
)

# #### Load model
longitudinales_model = ExtraSmallUNet(n_channels=3, n_classes=1)
longitudinales_model = torch.nn.DataParallel(longitudinales_model, device_ids=range(torch.cuda.device_count()))
longitudinales_model.load_state_dict(torch.load(model_longitudinales_path))
longitudinales_model.to(DEVICE)
longitudinales_model.eval()

# #### Inference
start_time = time.time()


@toma.batch(initial_batchsize=batch_size_severidades)
def run_longitudinales(batch_size):
    longitudinales_loader = DataLoader(
        longitudinales_dataset, batch_size=batch_size, pin_memory=True, shuffle=False
    )
    with torch.no_grad():
        for batch_idx, (inputs, _, paths) in enumerate(longitudinales_loader):
            inputs = inputs.cuda()
            outputs = longitudinales_model(inputs)

            y = torch.nn.Sigmoid()(outputs)
            masks = (y > 0.5).int()

            for indice, mask in enumerate(masks):
                path = paths[indice].split("/")[-1]
                mask = mask.cpu().numpy().squeeze()
                severidad = get_severidad_grieta(mask)

                labels_preds.loc[labels_preds["Image"] == path, "area_severidad_longitudinales"] = severidad[
                    "anchura_real"]
                labels_preds.loc[labels_preds["Image"] == path, "area_daño_longitudinales"] = severidad["longitud_real"]


run_longitudinales()
print("--- INFERENCIA LONGITUDINALES %s seconds ---" % (time.time() - start_time))

# ## Transversales

# #### Load Data
transversales_cases = labels_preds.loc[labels_preds["Grietas transversales"] == 1]
print(f"\nExisten {len(transversales_cases)} muestras con grietas transversales")
transversales_dataset = SIMEPU_Dataset_MultiLabel(
    from_df=transversales_cases, from_folder=DATA_DIR,
    augmentation=val_albumentation_severidades, fold=-1, get_path=True
)

# #### Load model
transversales_model = ExtraSmallUNet(n_channels=3, n_classes=1)
transversales_model = torch.nn.DataParallel(transversales_model, device_ids=range(torch.cuda.device_count()))
transversales_model.load_state_dict(torch.load(model_transversales_path))
transversales_model.to(DEVICE)
transversales_model.eval()

# #### Inference
start_time = time.time()


@toma.batch(initial_batchsize=batch_size_severidades)
def run_transversales(batch_size):
    transversales_loader = DataLoader(
        transversales_dataset, batch_size=batch_size, pin_memory=True, shuffle=False
    )
    with torch.no_grad():
        for batch_idx, (inputs, _, paths) in enumerate(transversales_loader):
            inputs = inputs.cuda()
            outputs = transversales_model(inputs)

            y = torch.nn.Sigmoid()(outputs)
            masks = (y > 0.5).int()

            for indice, mask in enumerate(masks):
                path = paths[indice].split("/")[-1]
                mask = mask.cpu().numpy().squeeze()
                severidad = get_severidad_grieta(mask)

                labels_preds.loc[labels_preds["Image"] == path, "area_severidad_transversales"] = severidad[
                    "anchura_real"]
                labels_preds.loc[labels_preds["Image"] == path, "area_daño_transversales"] = severidad["longitud_real"]


run_transversales()

print("--- INFERENCIA TRANSVERSALES %s seconds ---" % (time.time() - start_time))

# ## Cocodrilo

# #### Load Data
cocodrilo_cases = labels_preds.loc[labels_preds["Grietas en forma de piel de cocodrilo"] == 1]
print(f"\nExisten {len(cocodrilo_cases)} muestras con cocodrilo")
cocodrilo_dataset = SIMEPU_Dataset_MultiLabel(
    from_df=cocodrilo_cases, from_folder=DATA_DIR,
    augmentation=val_albumentation_severidades, fold=-1, get_path=True
)

# #### Inference
total_paths, total_areas = [], []
start_time = time.time()


@toma.batch(initial_batchsize=batch_size_severidades)
def run_cocodrilo(batch_size):
    cocodrilo_loader = DataLoader(
        cocodrilo_dataset, batch_size=batch_size, pin_memory=True, shuffle=False
    )
    with torch.no_grad():
        for batch_idx, (inputs, _, paths) in enumerate(cocodrilo_loader):
            inputs = inputs.cuda()
            outputs_l = longitudinales_model(inputs)
            outputs_t = transversales_model(inputs)

            y_l = torch.nn.Sigmoid()(outputs_l)
            y_t = torch.nn.Sigmoid()(outputs_t)
            masks_l = (y_l > 0.5).int()
            masks_t = (y_t > 0.5).int()

            masks = (torch.logical_or(masks_l, masks_t) * 1)

            total_paths.extend(paths)
            total_areas.extend([(mask.sum().item() / area_img) for mask in masks])


run_cocodrilo()

for path, area in zip(total_paths, total_areas):
    labels_preds.loc[labels_preds["Image"] == path.split("/")[-1], "area_severidad_cocodrilo"] = area
    labels_preds.loc[labels_preds["Image"] == path.split("/")[-1], "area_daño_cocodrilo"] = area

print("--- INFERENCIA COCODRILO %s seconds ---" % (time.time() - start_time))

print("\n--- INFERENCIA TOTAL %s seconds ---" % (time.time() - start_total_time))

labels_preds.to_csv("inference.csv", index=None)
