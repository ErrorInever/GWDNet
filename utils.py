import random
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
from config import cfg
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def visualise_series(img_path, id_series, target):
    """Visualise series"""
    SIGNAL_NAMES = ["LIGO Hanford", "LIGO Livingston", "Virgo"]
    COLORS = ['blue', 'red', 'green']
    x = np.load(img_path)
    plt.figure(figsize=(15,15))
    for i in range(3):
        plt.subplot(4, 1, i + 1)
        plt.plot(x[i], color=COLORS[i])
        plt.legend([SIGNAL_NAMES[i]], fontsize=12, loc="lower right")
        plt.subplot(4, 1, 4)
        plt.plot(x[i], color=COLORS[i])
    plt.subplot(4, 1, 4)
    plt.legend(SIGNAL_NAMES, fontsize=12, loc="lower right")
    plt.suptitle(f"ID: {id_series}, Target: {target}", fontsize=14)
    plt.show()


def show_batch(batch, title='None'):
    """Visualise batch"""
    series = batch[0]
    targets = batch[1]
    plt.figure(figsize=(15, 15))
    for i in range(series.shape[0]):
        plt.subplot(3, 1, i + 1)
        plt.plot(series[i])
        plt.gca().set_title(f"Signal №{i}, Target: {targets[i]}")
    plt.suptitle(title)
    plt.show()


def show_spectrogram(batch, transform, title='None'):
    """Show spectrogram"""
    series = batch[0]
    targets = batch[1]
    plt.figure(figsize=(15, 15))
    for i in range(series.shape[0]):
        plt.subplot(3, 1, i + 1)
        plt.pcolormesh(transform(series[i][0]).squeeze())
        plt.suptitle(f"Number: {i}, Target: {targets[i]}")
    plt.suptitle(title)
    plt.show()


def get_train_file_path(image_id):
    """Make path to train series"""
    return f'{cfg.DATA_FOLDER}/train/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.npy'


def get_test_file_path(image_id):
    """Make path to test series"""
    return f'{cfg.DATA_FOLDER}/test/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.npy'


def split_data_kfold(df, k=5):
    """
    Split data on part: Stratified K-Fold
    :param df: DataFrame object
    :param k: ``int``, How many folds the dataset is going to be divided
    :return: Divided DataFrame object
    """
    fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.SEED)
    for n, (train_idx, val_idx) in enumerate(fold.split(df, df['target'])):
        df.loc[val_idx, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    logger.info(f'==> Split K-Fold')
    logger.info(df.groupby(['fold', 'target']).size())
    return df


def get_scheduler(optimizer):
    """
    Define scheduler for train mode
    :param optimizer: ``torch.optim.Object``, train optimizer
    :return: instance of scheduler
    """
    if cfg.SCHEDULER_VERSION == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.FACTOR, patience=cfg.PATIENCE, verbose=True,
                                      eps=cfg.EPS)
    elif cfg.SCHEDULER_VERSION == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX, eta_min=cfg.MIN_LR, last_epoch=-1)
    elif cfg.SCHEDULER_VERSION == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, T_mult=1, eta_min=cfg.MIN_LR, last_epoch=-1)
    else:
        raise ValueError('SCHEDULER WAS NOT DEFINED')
    return scheduler


def save_checkpoint(save_path, model, optimizer, lr, preds):
    """
    Save state to hard drive
    :param save_path: ``str``, path to save state
    :param model: ``instance of nn.Module``, model
    :param optimizer: ``instance of optim.object``, optimizer
    :param lr: ``float``, current learning rate
    :param preds: ``List(floats)``, average eval loss of epoch, list of predictions
    """
    torch.save({
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'lr': lr,
        'preds': preds,
    }, save_path)


def print_result(result_df):
    """
    Display result of predictions
    :param result_df: ``DataFrame``, predictions
    """
    preds = result_df['preds'].values
    labels = result_df['target'].values
    score = roc_auc_score(labels, preds)
    logger.info(f'Score: {score:<.4f}')


def get_grad_cam(model, device, series, img, label):
    result = {"vis": None, "img": None, "prob": None, "label": None}

    with torch.no_grad():
        preds = model(series.unsqueeze(0).to(device))
    prob = np.concatenate(preds.sigmoid().to("cpu").numpy())[0]

    target_layer = model.model.conv_head
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)
    output = cam(input_tensor=series.unsqueeze(0))
    try:
        vis = show_cam_on_image(series.numpy().transpose((1, 2, 0)), output[0])
    except:
        return result

    result = {"vis": vis, "img": img, "prob": prob, "label": label}
    return result