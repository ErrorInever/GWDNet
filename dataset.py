import torch
import numpy as np
import albumentations as A
from scipy import signal
from torch.utils.data import Dataset
from nnAudio.Spectrogram import CQT1992v2
from config import cfg
from albumentations.pytorch import ToTensorV2


class GWDataset(Dataset):
    """Gravitation Wave Dataset"""
    def __init__(self, df, use_filter=True, use_transform=True, use_aug=True):
        self.df = df
        self.id_series = df['id'].values
        self.paths_series = df['img_path'].values
        self.targets = df['target'].values
        self.use_filter = use_filter
        self.wave_transform = CQT1992v2(**cfg.Q_TRANSFORM_PARAMS)
        self.transform = use_transform
        self.use_aug = use_aug

        if self.use_filter:
            self.bHP, self.aHP = signal.butter(8, (20, 500), btype='bandpass', fs=2048)

        self._aug = A.Compose([
            ToTensorV2()
        ])

        self._to_tensor = A.Compose([
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        :param idx: int
        :return: ``List(Tensor([N, C, H, W]), Tensor([N]))``
        """
        series = np.load(self.paths_series[idx])
        # Concatenate dimensions
        series = np.concatenate(series, axis=0)
        if self.use_filter:
            series *= signal.tukey(4096*3, 0.2)
            series = signal.filtfilt(self.bHP, self.aHP, series)
        # Normalize
        series = series / np.max(series)
        # Q-Transform
        if self.transform:
            series = torch.from_numpy(series).float()
            series = self.wave_transform(series).squeeze()
            series = series.squeeze().numpy()
        if self.use_aug:
            series = self._aug(image=series)['image']
        else:
            series = self._to_tensor(image=series)['image']
        targets = self.targets[idx]
        return series, torch.tensor(targets).float()


class GradCamGWDataset(Dataset):
    """Grad-cam helper dataset"""
    def __init__(self, df, use_filter=True, use_transform=True, use_aug=True):
        self.df = df
        self.id_series = df['id'].values
        self.paths_series = df['img_path'].values
        self.targets = df['target'].values
        self.use_filter = use_filter
        self.wave_transform = CQT1992v2(**cfg.Q_TRANSFORM_PARAMS)
        self.transform = use_transform
        self.use_aug = use_aug

        if self.use_filter:
            self.bHP, self.aHP = signal.butter(8, (20, 500), btype='bandpass', fs=2048)

        self._aug = A.Compose([
            ToTensorV2()
        ])

        self._to_tensor = A.Compose([
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        :param idx: int
        :return: ``List(Tensor([N, C, H, W]), Tensor([N]))``
        """
        series_id = self.id_series[idx]
        series = np.load(self.paths_series[idx])
        # Concatenate dimensions
        series = np.concatenate(series, axis=0)
        if self.use_filter:
            series *= signal.tukey(4096*3, 0.2)
            series = signal.filtfilt(self.bHP, self.aHP, series)
        # Normalize
        series = series / np.max(series)
        # Q-Transform
        if self.transform:
            series = torch.from_numpy(series).float()
            series = self.wave_transform(series).squeeze()
            series = series.squeeze().numpy()
        vis_series = series.copy()
        if self.use_aug:
            series = self._aug(image=series)['image']
        else:
            series = self._to_tensor(image=series)['image']
        targets = self.targets[idx]
        return series_id, series, vis_series, torch.tensor(targets).float()
