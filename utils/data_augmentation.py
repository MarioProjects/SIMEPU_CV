from torchvision import transforms
import albumentations


def get_augmentations(data_augmentation, pretrained, img_size, crop_size, segmentation_problem):
    train_aug, val_aug, train_albumentation, val_albumentation = None, None, None, None
    if not segmentation_problem:
        if data_augmentation:
            train_aug = [
                transforms.ToPILImage(),  # because the input dtype is numpy.ndarray
                transforms.Resize((img_size, img_size)),
                transforms.RandomCrop((crop_size, crop_size)),
                transforms.RandomHorizontalFlip(0.5),  # because this method is used for PIL Image dtype
                transforms.RandomVerticalFlip(0.5),  # because this method is used for PIL Image dtype
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),  # because inpus dtype is PIL Image
            ]
        else:
            train_aug = [
                transforms.ToPILImage(),  # because the input dtype is numpy.ndarray
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),  # because inpus dtype is PIL Image
            ]

        val_aug = [
            transforms.ToPILImage(),  # because the input dtype is numpy.ndarray
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),  # because inpus dtype is PIL Image
        ]

        if pretrained:
            train_aug.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            val_aug.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

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
