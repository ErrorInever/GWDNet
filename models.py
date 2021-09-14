import timm
import torch.nn as nn
from config import cfg


class EfficientNetVer(nn.Module):
    """EfficientNet b0-b7"""

    def __init__(self, version, pretrained=True):
        super().__init__()
        assert version in [0, 1, 2, 3, 4, 5, 6, 7], 'Efficient version must be int and in range 0-7'
        self.model = timm.create_model(f"tf_efficientnet_b{version}_ns", pretrained=pretrained, in_chans=1)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, cfg.NUM_CLASSES)

    def forward(self, x):
        return self.model(x)


def get_model(version=3, pretrained=True):
    return EfficientNetVer(version, pretrained=pretrained)
