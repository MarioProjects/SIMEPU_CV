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
from sklearn.model_selection import KFold
from skimage.exposure import match_histograms
import random


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
    def __init__(self, data_partition='', transform=None, augmentation=None, fold=0, selected_class="",
                 get_path=False, binary_problem=False, damaged_problem=False, segmentation_problem=False,
                 rotate=False, seed=42, data_mod="", normalization="reescale", histogram_matching=False):
        """
          - data_partition:
             -> If empty ("") returns all samples from TRAIN set
             -> If "train" returns corresponding 'fold' TRAIN set
             -> If "validation" returns corresponding  'fold' VALIDATION set
         - get_path: If we want to return image path (True) or not (False), debug purposes
        """
        if (fold + 1) > 5:
            assert False, f"Only 5 folds used on training (0,1,2,3,4), yours '{fold}'"
        if data_partition not in ["", "train", "validation", "segmentation_test"]:
            assert False, "Wrong data partition: {}".format(data_partition)
        if binary_problem and damaged_problem:
            assert False, "Please not binary_problem and damaged_problem at same time"

        self.SIMEPU_DATA_PATH = f"data/SIMEPU/{data_mod}"

        if binary_problem:
            data_paths = pd.read_csv(f"utils/{data_mod}data_damages_path.csv")
            self.num_classes = 1
        elif damaged_problem:
            data_paths = pd.read_csv(f"utils/{data_mod}only_damaged_path.csv")
            self.num_classes = 6
        else:
            data_paths = pd.read_csv(f"utils/{data_mod}data_paths.csv")
            self.num_classes = len(np.unique(data_paths["target"]))

        if selected_class != "":
            data_paths = data_paths[data_paths.path.str.startswith(selected_class)]
            self.num_classes = 1

        if segmentation_problem:
            if selected_class == "":
                assert False, "You need select a class for segmentation problem"
            if os.path.isdir(os.path.join(self.SIMEPU_DATA_PATH, "Mascaras", selected_class)):
                masks_paths = os.listdir(os.path.join(self.SIMEPU_DATA_PATH, "Mascaras", selected_class))
            else:
                masks_paths = []
            for index, row in data_paths.iterrows():
                if data_partition == "segmentation_test":  # Not labeled images
                    if row['path'].split("/")[1] in masks_paths:
                        data_paths.drop(index, inplace=True)
                elif row['path'].split("/")[1] not in masks_paths:
                    data_paths.drop(index, inplace=True)

        self.data_paths = data_paths
        if data_partition == "" or data_partition == "segmentation_test":
            self.data_paths = self.data_paths
        elif fold != -1:
            kf = KFold(n_splits=5, random_state=seed, shuffle=True)
            for fold_number, (train_index, val_index) in enumerate(kf.split(data_paths)):
                if fold_number == fold:
                    if data_partition == "train":
                        self.data_paths = self.data_paths.iloc[train_index]
                    elif data_partition == "validation":
                        self.data_paths = self.data_paths.iloc[val_index]
                    else:
                        assert False, "Wrong data partition: {}".format(data_partition)
                    break

        self.data_paths = self.data_paths.reset_index(drop=True)
        self.data_partition = data_partition
        self.transform = transform
        self.augmentation = augmentation
        self.get_path = get_path or segmentation_problem
        self.segmentation_problem = segmentation_problem
        self.selected_class = selected_class
        self.rotate = rotate
        self.normalization = normalization

        self.histogram_matching = histogram_matching
        if self.histogram_matching:
            self.hist_data_mod = "" if data_mod != "" else data_mod  # Load opposite data
            self.hist_match_df = pd.read_csv(f"utils/{self.hist_data_mod}data_paths.csv")

    def __getitem__(self, idx):

        img_path = self.SIMEPU_DATA_PATH + "/" + self.data_paths.iloc[idx]["path"]
        img = io.imread(img_path)
        target = int(self.data_paths.iloc[idx]["target"])
        if self.data_partition == "train" and self.histogram_matching and (
                random.random() < 0.33):  # happens 33% of the time
            if "retrain" in self.SIMEPU_DATA_PATH:
                base_path = self.SIMEPU_DATA_PATH[:-len("retrainvX")]  # remove retrainvX part
            else:
                base_path = os.path.join(self.SIMEPU_DATA_PATH, self.hist_data_mod)
            rand_hist_img_path = os.path.join(base_path, self.hist_match_df.sample(n=1).iloc[0]["path"])
            rand_hist_img = io.imread(rand_hist_img_path)
            img = match_histograms(img, rand_hist_img, multichannel=True)
        original_img = copy.deepcopy(img)
        mask = None

        if self.transform is not None:
            img = transforms.Compose(self.transform)(img)
        elif self.augmentation is not None:
            if self.data_partition == "segmentation_test":
                img, mask = apply_augmentations(img, albumentations.Compose(self.augmentation), None, None)
                img = apply_normalization(img, self.normalization)
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # Transpose == Channels first
            else:
                mask_path = os.path.join(self.SIMEPU_DATA_PATH, "Mascaras", self.data_paths.iloc[idx]["path"])
                mask = np.where(io.imread(mask_path)[..., 0] > 0.5, 1, 0).astype(np.int32)
                original_mask = copy.deepcopy(mask)
                if self.rotate:
                    # Debemos voltear la mascara de grietas transversales ya que la imagen es rotada
                    original_img = albumentations.Rotate(limit=(90, 90), p=1)(image=original_img)["image"]
                    original_mask = albumentations.Rotate(limit=(90, 90), p=1)(image=original_mask.astype(np.uint8))[
                        "image"]
                img, mask = apply_augmentations(img, albumentations.Compose(self.augmentation), None, mask)
                img = apply_normalization(img, self.normalization)
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


class SIMEPU_Segmentation_Dataset(data.Dataset):
    def __init__(self, data_partition='', transform=None, augmentation=None, fold=0, selected_class="",
                 rotate=False, seed=2, data_mod="", normalization="reescale"):
        """
          - data_partition:
             -> If empty ("") returns all samples from TRAIN set
             -> If "train" returns corresponding 'fold' TRAIN set
             -> If "validation" returns corresponding  'fold' VALIDATION set
         - get_path: If we want to return image path (True) or not (False), debug purposes
        """
        if (fold + 1) > 5:
            assert False, f"Only 5 folds used on training (0,1,2,3,4), yours '{fold}'"
        if data_partition not in ["", "train", "validation", "segmentation_test"]:
            assert False, "Wrong data partition: {}".format(data_partition)

        self.SIMEPU_DATA_PATH = f"data/SIMEPU/{data_mod}"
        self.num_classes = 1

        if selected_class == "":
            assert False, "You need select a class for segmentation problem"

        data_paths = []
        for subdir, dirs, files in os.walk(self.SIMEPU_DATA_PATH):
            for file in files:
                file_path = os.path.join(subdir, file)
                if f"/masks/{selected_class}" in file_path:
                    data_paths.append(file_path)

        self.data_paths = np.array(data_paths)
        if data_partition == "" or data_partition == "segmentation_test":
            self.data_paths = self.data_paths
        elif fold != -1:
            kf = KFold(n_splits=5, random_state=seed, shuffle=True)
            for fold_number, (train_index, val_index) in enumerate(kf.split(data_paths)):
                if fold_number == fold:
                    if data_partition == "train":
                        self.data_paths = self.data_paths[train_index]
                    elif data_partition == "validation":
                        self.data_paths = self.data_paths[val_index]
                    else:
                        assert False, "Wrong data partition: {}".format(data_partition)
                    break

        self.data_partition = data_partition
        self.transform = transform
        self.augmentation = augmentation
        self.selected_class = selected_class
        self.rotate = rotate
        self.normalization = normalization

    def __getitem__(self, idx):

        img_path = self.data_paths[idx]
        img_path = img_path[:img_path.find("masks")] + "images" + img_path[img_path.find("/VIRB"):][:-3] + "jpg"
        img = io.imread(img_path)
        target = 0  # Unnecesary
        original_img = copy.deepcopy(img)
        mask, original_mask = None, None

        if self.transform is not None:
            img = transforms.Compose(self.transform)(img)
        elif self.augmentation is not None:
            if self.data_partition == "segmentation_test":
                img, mask = apply_augmentations(img, albumentations.Compose(self.augmentation), None, None)
                img = apply_normalization(img, self.normalization)
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # Transpose == Channels first
            else:
                mask_path = self.data_paths[idx]
                mask = np.where(io.imread(mask_path)[..., 0] > 0.5, 1, 0).astype(np.int32)
                original_mask = copy.deepcopy(mask)
                if self.rotate:
                    # Debemos voltear la mascara de grietas transversales ya que la imagen es rotada
                    original_img = albumentations.Rotate(limit=(90, 90), p=1)(image=original_img)["image"]
                    original_mask = albumentations.Rotate(limit=(90, 90), p=1)(image=original_mask.astype(np.uint8))[
                        "image"]
                img, mask = apply_augmentations(img, albumentations.Compose(self.augmentation), None, mask)
                img = apply_normalization(img, self.normalization)
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # Transpose == Channels first
                mask = torch.from_numpy(np.expand_dims(mask, 0)).float()

        res = [img, target]

        if self.data_partition != "segmentation_test":
            res = res + [mask, original_img, original_mask]
        else:
            res = res + [original_img]

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


class SIMEPU_Dataset_MultiLabel(data.Dataset):
    def __init__(self, data_partition='', transform=None, augmentation=None, fold=0, selected_class="",
                 get_path=False, segmentation_problem=False, rotate=False, seed=42,
                 normalization="reescale"):
        """
          - data_partition:
             -> If empty ("") returns all samples from TRAIN set
             -> If "train" returns corresponding 'fold' TRAIN set
             -> If "validation" returns corresponding  'fold' VALIDATION set
         - get_path: If we want to return image path (True) or not (False), debug purposes
        """
        if (fold + 1) > 5:
            assert False, f"Only 5 folds used on training (0,1,2,3,4), yours '{fold}'"
        if data_partition not in ["", "train", "validation", "segmentation_test"]:
            assert False, "Wrong data partition: {}".format(data_partition)

        self.SIMEPU_DATA_PATH = "data/SIMEPU/"

        df0 = pd.read_csv(os.path.join(self.SIMEPU_DATA_PATH, "v0", "info.csv"))
        df0['Image'] = 'v0/images/' + df0['Image'].astype(str)
        df1 = pd.read_csv(os.path.join(self.SIMEPU_DATA_PATH, "v1", "info.csv"))
        df1['Image'] = 'v1/images/' + df1['Image'].astype(str)
        df2 = pd.read_csv(os.path.join(self.SIMEPU_DATA_PATH, "v2", "info.csv"))
        df2['Image'] = 'v2/images/' + df2['Image'].astype(str)
        df3 = pd.read_csv(os.path.join(self.SIMEPU_DATA_PATH, "v3", "info.csv"))
        df3['Image'] = 'v3/images/' + df3['Image'].astype(str)

        df = [df0, df1, df2, df3]
        df = pd.concat(df)
        df = df.reset_index(drop=True)
        initial_df = df.copy()
        aux_df = None

        self.classes = [
            "Alcantarillado", "Marca vial", "Hueco", "Parcheo", "Grietas longitudinales",
            "Grietas en forma de piel de cocodrilo", "Grietas transversales", "Meteorización y desprendimiento"
        ]
        self.num_classes = len(self.classes)

        if segmentation_problem:
            self.num_classes = 1
            if selected_class == "":
                assert False, "You need select a class for segmentation problem"
            if data_partition != "segmentation_test":
                df = df[df[f'Mask {selected_class}'].notna()]

        if data_partition == "segmentation_test":
            # Take all samples with selected class and not specified mask
            aux_df = initial_df[(initial_df[selected_class] == 1) & (pd.isnull(initial_df[f"Mask {selected_class}"]))]

        if fold != -1 and data_partition != "":
            kf = KFold(n_splits=5, random_state=seed, shuffle=True)
            for fold_number, (train_index, val_index) in enumerate(kf.split(df)):
                if fold_number == fold:
                    if data_partition == "train":
                        df = df.iloc[train_index]
                    elif data_partition == "validation" or data_partition == "segmentation_test":
                        df = df.iloc[val_index]
                    else:
                        assert False, "Wrong data partition: {}".format(data_partition)
                    break

        df = df if aux_df is None else pd.concat([df, aux_df])

        df = df.reset_index(drop=True)
        self.data_paths = df
        self.data_partition = data_partition
        self.transform = transform
        self.augmentation = augmentation
        self.get_path = get_path or segmentation_problem
        self.segmentation_problem = segmentation_problem
        self.selected_class = selected_class
        self.rotate = rotate
        self.normalization = normalization

    def __getitem__(self, idx):

        img_path = os.path.join(self.SIMEPU_DATA_PATH, self.data_paths.iloc[idx]["Image"])
        version = self.data_paths.iloc[idx]["Image"][0:2]
        img = io.imread(img_path)
        original_img = copy.deepcopy(img)
        target = np.expand_dims(np.array(self.data_paths.iloc[idx][self.classes].tolist()), axis=0)
        mask = None

        if self.transform is not None:
            img = transforms.Compose(self.transform)(img)
        elif self.augmentation is not None:

            if self.segmentation_problem and self.data_partition != "segmentation_test":
                mask_path = os.path.join(self.SIMEPU_DATA_PATH, version,
                                         self.data_paths.iloc[idx][f"Mask {self.selected_class}"])
                mask = np.where(io.imread(mask_path)[..., 0] > 0.5, 1, 0).astype(np.int32)
                original_mask = copy.deepcopy(mask)

            if self.rotate:
                # Debemos voltear la mascara de grietas transversales ya que la imagen es rotada
                original_img = albumentations.Rotate(limit=(90, 90), p=1)(image=original_img)["image"]
                original_mask = albumentations.Rotate(limit=(90, 90), p=1)(image=original_mask.astype(np.uint8))[
                    "image"]

            img, mask = apply_augmentations(img, albumentations.Compose(self.augmentation), None, mask)
            img = apply_normalization(img, self.normalization)
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # Transpose == Channels first
            mask = torch.from_numpy(np.expand_dims(mask, 0)).float() if mask is not None else None

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
