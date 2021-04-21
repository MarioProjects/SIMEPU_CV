from torchvision import transforms
import albumentations
from PIL import Image
import numpy as np
import torch

from RandAugment.augmentations import Lighting, RandAugment

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ]
}


# https://github.com/ildoonet/pytorch-randaugment
def get_augmentations(pretrained, img_size, segmentation_problem, randaug_n, randaug_m, cutout_size):
    train_aug, val_aug, train_albumentation, val_albumentation = None, None, None, None
    if not segmentation_problem:
        train_aug = transforms.Compose([
            transforms.ToPILImage(),  # because the input dtype is numpy.ndarray
            transforms.Resize(img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        val_aug = transforms.Compose([
            transforms.ToPILImage(),  # because the input dtype is numpy.ndarray
            transforms.Resize(img_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        if pretrained:
            #train_aug.transforms.append(Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']), )
            train_aug.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            val_aug.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        if randaug_n > 0 and randaug_m > 0:
            train_aug.transforms.insert(1, RandAugment(randaug_n, randaug_m))

        if cutout_size > 0:
            train_aug.transforms.append(CutoutDefault(cutout_size))

    else:  # Segmentation problem
        train_albumentation = [
            albumentations.Resize(img_size, img_size),
            albumentations.ElasticTransform(p=0.72, alpha=177, sigma=177 * 0.05, alpha_affine=176 * 0.03),
            albumentations.GridDistortion(p=0.675, distort_limit=0.3),
            albumentations.OpticalDistortion(p=0.2, distort_limit=0.2, shift_limit=0.2),

            albumentations.ShiftScaleRotate(p=0.56, shift_limit=0.2, scale_limit=0.0, rotate_limit=0),  # shift
            albumentations.ShiftScaleRotate(p=0.25, shift_limit=0.0, scale_limit=0.2, rotate_limit=0),  # scale

            albumentations.VerticalFlip(p=0.325),
            albumentations.HorizontalFlip(p=0.3),
        ]

        val_albumentation = [
            albumentations.Resize(img_size, img_size),
        ]

    return train_aug, val_aug, train_albumentation, val_albumentation


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
