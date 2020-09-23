from collections import OrderedDict
import numpy as np
import cv2
import torch
import albumentations


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
