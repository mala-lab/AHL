from torch.utils.data import Dataset
import numpy as np
import torch

class BaseADDataset(Dataset):
    def __init__(self):
        super(BaseADDataset).__init__()

        self.normal_idx = None
        self.outlier_idx = None

class Task_Dataset(Dataset):
    def __init__(self, feature, feature_s, label):
        super(Task_Dataset).__init__()
        self.feature = feature
        self.feature_s = feature_s
        self.label = label
        self.normal_idx = np.argwhere(self.label == 0).flatten()
        self.outlier_idx = np.argwhere(self.label == 1).flatten()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sample = {"image": self.feature[index], "image_scale": self.feature_s[index], "label": self.label[index]}
        return sample