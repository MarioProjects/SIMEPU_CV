from collections import OrderedDict
import cv2

import os
import albumentations
import torch
import numpy as np
import pandas as pd
from skimage import io
import copy
from torch.utils import data
from sklearn.model_selection import KFold


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


def load_dataparallel_model(model, checkpoint):
    new_state_dict = OrderedDict()

    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model


def load_predict_segmentation(model, model_checkpoint, img, batch, DEVICE):
    model = load_dataparallel_model(
        model, torch.load(model_checkpoint)
    )
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(batch.to(DEVICE))
    original_h, original_w, _ = img.shape
    for indx, single_pred in enumerate(outputs):
        resize_transform = albumentations.Resize(original_h, original_w)
        pred_mask = resize_transform(image=torch.sigmoid(single_pred).squeeze(0).data.cpu().numpy())["image"]
        binary_pred_mask = np.where(pred_mask > 0.5, 1, 0).astype(np.int32)
        break
    kernel = np.ones((4, 4), np.uint8)
    erosion = cv2.erode(np.uint8(binary_pred_mask), kernel, iterations=1)

    mask = erosion > 0  # or `x != 255` where x is your array
    return mask


class SIMEPU_Dataset_MultiLabel(data.Dataset):
    def __init__(self, data_partition='', transform=None, augmentation=None, fold=0, selected_class="",
                 get_path=False, segmentation_problem=False, rotate=False, seed=42,
                 normalization="reescale", from_folder="", from_df=None):
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

        self.from_folder = from_folder
        if from_folder == "":
            self.SIMEPU_DATA_PATH = "data/SIMEPU/"

            df = pd.read_csv(os.path.join(self.SIMEPU_DATA_PATH, "info.csv"))
            df['Image'] = 'images/' + df['Image'].astype(str)
        elif from_df is not None:
            df = from_df
            self.SIMEPU_DATA_PATH = from_folder
        else:
            self.SIMEPU_DATA_PATH = from_folder
            df = pd.DataFrame({"Image": os.listdir(from_folder)})

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
        img = io.imread(img_path)
        original_img = copy.deepcopy(img)
        if self.from_folder != "":
            target = 0
        else:
            target = np.expand_dims(np.array(self.data_paths.iloc[idx][self.classes].tolist()), axis=0)
        mask = None

        if self.transform is not None:
            img = self.transform(img)
        elif self.augmentation is not None:

            if self.segmentation_problem and self.data_partition != "segmentation_test":
                mask_path = os.path.join(self.SIMEPU_DATA_PATH, "masks",
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
