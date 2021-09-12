import os
import random
import logging
import torch
import numpy as np


def seed_everything(seed):
    """
    Seed everything
    :param seed: ``int``, seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True