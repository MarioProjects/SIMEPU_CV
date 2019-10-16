import os
import pickle

import numpy as np
import pandas as pd
from skimage import io
from torch.utils import data
from torch.utils.data import DataLoader

# echo "export SIMEPU_DATA='/home/maparla/DeepLearning/Datasets/SIMEPU'" >> ~/.bashrc
if os.environ.get('SIMEPU_DATA') is not None:
    SIMEPU_DATA_PATH = os.environ.get('SIMEPU_DATA')
else:
    assert False, "Please set SIMEPU DATA Path environment variable!"

SIMEPU_PATHS = pd.read_csv(os.path.join(SIMEPU_DATA_PATH, "data_paths.csv"))

with open(os.path.join(SIMEPU_DATA_PATH, "labels2targets.pkl"), 'rb') as f:
    LABELS2TARGETS = pickle.load(f)
with open(os.path.join(SIMEPU_DATA_PATH, "targets2labels.pkl"), 'rb') as f:
    TARGETS2LABELS = pickle.load(f)


class SIMEPU_Dataset(data.Dataset):
    def __init__(self, data_partition='', transform=None, validation_size=0.15, seed=42, get_path=False):
        """
          - data_partition:
             -> Si esta vacio ("") devuelve todas las muestras del TRAIN set
             -> Si es "train" devuelve '1-validation_size' muestras del TRAIN set
             -> Si es "validation" devuelve 'validation_size' muestras del TRAIN set
         - get_path: Si queremos devolver el path de la imagen (True) o no (False), tipicamente depuracion
        """

        np.random.seed(seed=seed)

        if data_partition == "":
            self.data_paths = SIMEPU_PATHS
        else:
            msk = np.random.rand(len(SIMEPU_PATHS)) < validation_size
            if data_partition == "train":
                self.data_paths = SIMEPU_PATHS[msk]
            elif data_partition == "validation":
                self.data_paths = SIMEPU_PATHS[~msk]
            else:
                assert False, "Wrong data partition: {}".format(data_partition)

        self.data_partition = data_partition
        self.transform = transform
        self.get_path = get_path

    def __getitem__(self, idx):
        img_path = self.data_paths.iloc[idx]["path"]
        target = int(self.data_paths.iloc[idx]["target"])
        img = io.imread(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.get_path:
            return img, target, img_path
        else:
            return img, target

    def __len__(self):
        return len(self.data_paths)


def test():
    train_dataset = SIMEPU_Dataset(data_partition="train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    xbatch, ybatch = next(iter(train_loader))
    print("Batch shape: {} / Min: {:.2f} / Max: {:.4f}".format(xbatch.shape, xbatch.min(), xbatch.max()))

# test()
