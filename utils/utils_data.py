import os
import albumentations
import pickle
import torch
import numpy as np
import pandas as pd
from skimage import io
import copy
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

# echo "export SIMEPU_DATA='/home/maparla/DeepLearning/Datasets/SIMEPU'" >> ~/.bashrc
if os.environ.get('SIMEPU_DATA') is not None:
    SIMEPU_DATA_PATH = os.environ.get('SIMEPU_DATA')
else:
    assert False, "Please set SIMEPU_DATA environment variable!"

with open("utils/labels2targets.pkl", 'rb') as f:
    LABELS2TARGETS = pickle.load(f)
with open("utils/targets2labels.pkl", 'rb') as f:
    TARGETS2LABELS = pickle.load(f)

with open("utils/labels2targetsdamaged.pkl", 'rb') as f:
    LABELS2TARGETSDAMAGED = pickle.load(f)
with open("utils/targets2labelsdamaged.pkl", 'rb') as f:
    TARGETS2LABELSDAMAGED = pickle.load(f)


def apply_augmentations(image, transform, img_transform, mask=None):
    if transform is not None:
        if mask is not None:
            augmented = transform(image=image, mask=mask)
            mask = augmented['mask']
        else:
            augmented = transform(image=image)

        image = augmented['image']

    if img_transform is not None:
        augmented = img_transform(image=image)
        image = augmented['image']

    return image, mask


def apply_normalization(image, normalization_type):
    """
    https://www.statisticshowto.com/normalized/
    :param image:
    :param normalization_type:
    :return:
    """
    if normalization_type == "none":
        return image
    elif normalization_type == "reescale":
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)
        return image
    elif normalization_type == "standardize":
        mean = np.mean(image)
        std = np.std(image)
        image = image - mean
        image = image / std
        return image
    assert False, "Unknown normalization: '{}'".format(normalization_type)


"""
 - Las clases SIN daño: [1] Marca Via / [2] Sin Daño / [8] Alcantarillado
 - Las clases CON daño: 
       [0] Parcheo / [3] Grietas Transversales Daño / [4] Huecos / [5] Grietas longitudinales / 
       [6] Meteorización y desprendimiento / [7] Grietas en forma de piel de cocodrilo
"""


class SIMEPU_Dataset(data.Dataset):
    def __init__(self, data_partition='', transform=None, augmentation=None, validation_size=0.15, selected_class="",
                 get_path=False, binary_problem=False, damaged_problem=False, segmentation_problem=False, seed=42):
        """
          - data_partition:
             -> Si esta vacio ("") devuelve todas las muestras del TRAIN set
             -> Si es "train" devuelve '1-validation_size' muestras del TRAIN set
             -> Si es "validation" devuelve 'validation_size' muestras del TRAIN set
         - get_path: Si queremos devolver el path de la imagen (True) o no (False), tipicamente depuracion
         - dano_no_dano: Para separa las muestras en daño / no daño
        """
        if data_partition not in ["", "train", "validation", "segmentation_test"]:
            assert False, "Wrong data partition: {}".format(data_partition)
        if binary_problem and damaged_problem:
            assert False, "Please not binary_problem and damaged_problem at same time"

        if binary_problem:
            data_paths = pd.read_csv("utils/data_damages_path.csv")
            self.num_classes = 1
        elif damaged_problem:
            data_paths = pd.read_csv("utils/only_damaged_path.csv")
            self.num_classes = 6
        else:
            data_paths = pd.read_csv("utils/data_paths.csv")
            self.num_classes = len(np.unique(data_paths["target"]))

        if selected_class != "":
            data_paths = data_paths[data_paths.path.str.startswith(selected_class)]
            self.num_classes = 1

        if segmentation_problem:
            if selected_class == "": assert False, "You need select a class for segmentation problem"
            if os.path.isdir(os.path.join(SIMEPU_DATA_PATH, "Mascaras", selected_class)):
                masks_paths = os.listdir(os.path.join(SIMEPU_DATA_PATH, "Mascaras", selected_class))
            else: 
                masks_paths = []
            for index, row in data_paths.iterrows():
                if data_partition == "segmentation_test": # Aquellas imagenes que no tenemos etiquetadas
                    if row['path'].split("/")[1] in masks_paths:
                        data_paths.drop(index, inplace=True)
                elif row['path'].split("/")[1] not in masks_paths:
                    data_paths.drop(index, inplace=True)

        np.random.seed(seed=seed)
        if data_partition == "" or data_partition == "segmentation_test":
            self.data_paths = data_paths
        else:
            msk = np.random.rand(len(data_paths)) < validation_size
            if data_partition == "train":
                self.data_paths = data_paths[~msk]
            elif data_partition == "validation":
                self.data_paths = data_paths[msk]
            else:
                assert False, "Wrong data partition: {}".format(data_partition)

        self.data_paths = self.data_paths.reset_index(drop=True)
        self.data_partition = data_partition
        self.transform = transform
        self.augmentation = augmentation
        self.get_path = get_path or segmentation_problem
        self.segmentation_problem = segmentation_problem

    def __getitem__(self, idx):

        img_path = SIMEPU_DATA_PATH + "/" + self.data_paths.iloc[idx]["path"]
        target = int(self.data_paths.iloc[idx]["target"])
        img = io.imread(img_path)
        original_img = copy.deepcopy(img)
        mask = None

        if self.transform is not None:
            img = transforms.Compose(self.transform)(img)
        elif self.augmentation is not None:
            if self.data_partition == "segmentation_test":
                img, mask = apply_augmentations(img, albumentations.Compose(self.augmentation), None, None)
                img = apply_normalization(img, "reescale")
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # Transpose == Channels first
            else:
                mask_path = os.path.join(SIMEPU_DATA_PATH, "Mascaras", self.data_paths.iloc[idx]["path"])
                mask = np.where(io.imread(mask_path)[..., 0] > 0.5, 1, 0).astype(np.int32)
                original_mask = copy.deepcopy(mask)
                img, mask = apply_augmentations(img, albumentations.Compose(self.augmentation), None, mask)
                img = apply_normalization(img, "reescale")
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # Transpose == Channels first
                mask = torch.from_numpy(np.expand_dims(mask, 0)).float()

        res = [img, target]

        if self.segmentation_problem and self.data_partition != "segmentation_test":
            res = res + [mask, original_img, original_mask]
        if self.data_partition == "segmentation_test":
            res = res + [original_img]

        if self.get_path:
            res.append(img_path)

        return res

    def __len__(self):
        return len(self.data_paths)

    def segmentation_collate(self, batch):
        """
        Necesitamos un collate ya que las imagenes 'originales' pueden venir con distinto tamaño y no le gusta a Pytorch
        """
        img, target, mask, original_img, original_mask, img_path = [], [], [], [], [], []
        for item in batch:
            img.append(item[0])
            target.append(item[1])
            mask.append(item[2])
            original_img.append(item[3])
            original_mask.append(item[4])
            img_path.append(item[5])
        img = torch.stack(img)
        mask = torch.stack(mask)
        return img, target, mask, original_img, original_mask, img_path


def test():
    train_dataset = SIMEPU_Dataset(data_partition="train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    xbatch, ybatch = next(iter(train_loader))
    print("Batch shape: {} / Min: {:.2f} / Max: {:.4f}".format(xbatch.shape, xbatch.min(), xbatch.max()))

# test()
