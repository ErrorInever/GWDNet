import torch
import numpy as np
from scipy import signal
from torch.utils.data import Dataset
from nnAudio.Spectrogram import CQT1992v2
from config import cfg


class GWDataset(Dataset):
    """Gravitation Wave Dataset"""
    def __init__(self, df, use_filter=True, use_transform=True):
        self.df = df
        self.id_series = df['id'].values
        self.paths_series = df['img_path'].values
        self.targets = df['target'].values
        self.use_filter = use_filter
        self.wave_transform = CQT1992v2(**cfg.Q_TRANSFORM_PARAMS)
        self.transform = use_transform

        if self.use_filter:
            self.bHP, self.aHP = signal.butter(8, (20, 500), btype='bandpass', fs=2048)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
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
        targets = self.targets[idx]
        return torch.tensor(series, dtype=torch.float), torch.tensor(targets, dtype=torch.long)
