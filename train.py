import os
import argparse
import time
import warnings
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from utils import *
from config import cfg
from dataset import GWDataset, GradCamGWDataset
from torch.utils.data import DataLoader
from models import get_model
from metrics import MetricLogger
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Gravitation wave detection')
    parser.add_argument('--data_path', dest='data_path', help='Path to root dataset', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--device', dest='device', help='Use device: gpu or cpu. Default use gpu if available',
                        default='gpu', type=str)
    parser.add_argument('--eff_ver', dest='eff_ver', help='Efficient version', default=4, type=int)
    parser.add_argument('--num_folds', dest='num_folds', help='Number of K folds', default=5, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Number of epochs', default=1, type=int)
    parser.add_argument('--ckpt', dest='ckpt', help='Path to model ckpt.pth.tar', default=None, type=str)
    parser.add_argument('--run_name', dest='run_name', help='Run name of wandb', default=None, type=str)
    parser.add_argument('--wandb_id', dest='wandb_id', help='Wand metric id for resume train', default=None, type=str)
    parser.add_argument('--wandb_key', dest='wandb_key', help='Use this option if you run this from kaggle notebook, '
                                                              'input api key wandb', default=None, type=str)
    parser.add_argument('--test', dest='test', help='Test code on one fold and one epoch', action='store_true')

    parser.print_help()
    return parser.parse_args()


def train_one_epoch(model, optimizer, criterion, dataloader, metric_logger, device):
    """
    Train one epoch
    :param model: ``instance of nn.Module``, model
    :param optimizer: ``instance of optim.object``, optimizer
    :param criterion: ``nn.Object``, loss function
    :param dataloader: ``instance of Dataloader``, dataloader on train data
    :param metric_logger: ``instance of MetricLogger``, helper class
    :param device: ``str``, cpu or gpu
    :return: ``float``, average loss on epoch
    """
    model.train()
    losses = AverageMeter()

    if cfg.USE_APEX:
        scaler = GradScaler()

    loop = tqdm(dataloader, leave=True)
    for batch_idx, (img, label) in enumerate(loop):
        batch_size = label.size(0)
        img = img.to(device)
        label = label.to(device)

        if cfg.USE_APEX:
            with torch.cuda.amp.autocast():
                y_preds = model(img)
                loss = criterion(y_preds.view(-1), label)
        else:
            y_preds = model(img)
            loss = criterion(y_preds.view(-1), label)

        losses.update(loss.item(), batch_size)

        if cfg.USE_APEX:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

        if batch_idx % cfg.LOSS_FREQ == 0 or batch_idx == (len(dataloader)-1):
            metric_logger.train_loss(losses.val)

        loop.set_postfix(loss=losses.val)

    return losses.avg


def evaluate(model, criterion, dataloader, metric_logger, device):
    """
    Evaluate one epoch
    :param model: ``instance of nn.Module``, model
    :param criterion: ``nn.Object``, loss function
    :param dataloader: ``instance of Dataloader``, dataloader on train data
    :param metric_logger: ``instance of MetricLogger``, helper class
    :param device: ``str``, cpu or gpu
    :return: ``List([float, list])``, average loss of epoch, list predictions
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    loop = tqdm(dataloader, leave=True)
    for batch_idx, (img, label) in enumerate(loop):
        batch_size = label.size(0)
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            y_preds = model(img)
        loss = criterion(y_preds.view(-1), label)
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to('cpu').numpy())

        if batch_idx % cfg.LOSS_FREQ == 0 or batch_idx == (len(dataloader)-1):
            metric_logger.val_loss(losses.val)

        loop.set_postfix(loss=losses.val)

    predictions = np.concatenate(preds)
    return losses.avg, predictions


if __name__ == '__main__':
    seed_everything(cfg.SEED)
    args = parse_args()
    assert args.data_path, 'Data path not specified'
    assert args.device in ['gpu', 'cpu'], 'Incorrect device type'
    assert args.eff_ver in [0, 1, 2, 3, 4, 5, 6, 7], 'Efficient version must be int and in range 0-7'

    cfg.DATA_FOLDER = args.data_path
    logger = logging.getLogger('Train')

    if args.out_dir:
        cfg.OUTPUT_DIR = args.out_dir
    if args.device == 'gpu':
        cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cpu':
        cfg.DEVICE = 'cpu'
    if args.run_name:
        cfg.RUN_NAME = args.run_name
    if args.wandb_id:
        cfg.RESUME_ID = args.wandb_id
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
    if args.eff_ver:
        cfg.EFF_VER = args.eff_ver
        cfg.MODEL_TYPE = f"tf_efficientnet_b{cfg.EFF_VER}_ns"

    if args.num_folds:
        cfg.NUM_FOLDS = args.num_folds
    if args.num_epochs:
        cfg.NUM_EPOCHS = args.num_epochs

    if args.test:
        cfg.NUM_EPOCHS = 1
        cfg.NUM_FOLDS = 10 # for fast result

    logger.info(f'==> Start {__name__} at {time.ctime()}')
    logger.info(f'==> Called with args: {args.__dict__}')
    logger.info(f'==> Config params: {cfg.__dict__}')
    logger.info(f'==> Using device: {args.device}')
    logger.info(f'==> Model version: {args.eff_ver}')

    # Paths
    train_path = os.path.join(cfg.DATA_FOLDER, 'train')
    test_path = os.path.join(cfg.DATA_FOLDER, 'test')
    sub_path = os.path.join(cfg.DATA_FOLDER, 'sample_submission.csv')
    labels_path = os.path.join(cfg.DATA_FOLDER, 'training_labels.csv')
    # DATA
    sample_sub = pd.read_csv(sub_path)
    train_labels = pd.read_csv(labels_path)
    train_df = pd.read_csv(labels_path)
    train_df['img_path'] = train_df['id'].apply(get_train_file_path)
    test_df = pd.read_csv(sub_path)
    test_df['img_path'] = test_df['id'].apply(get_test_file_path)

    # Cross validation split to K folds
    train_df = split_data_kfold(train_df, cfg.NUM_FOLDS)
    oof_df = pd.DataFrame()
    for fold in range(cfg.NUM_FOLDS):
        logger.info(f'========== Fold: [{fold + 1} of {len(cfg.FOLD_LIST)}] ==========')
        # Each fold divide on train and validation datasets
        train_idxs = train_df[train_df['fold'] != fold].index
        val_idxs = train_df[train_df['fold'] == fold].index
        train_folds = train_df.loc[train_idxs].reset_index(drop=True)
        val_folds = train_df.loc[val_idxs].reset_index(drop=True)
        val_labels = val_folds['target'].values  # list of validation dataset targets of current fold

        train_dataset = GWDataset(train_df, use_filter=True, use_transform=True, use_aug=True)
        val_dataset = GWDataset(val_folds, use_filter=True, use_transform=True, use_aug=False)

        train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                      num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                                    pin_memory=True)
        # Define model & optimizer & scheduler & loss function
        # TODO Load checkpoint
        model = get_model(cfg.EFF_VER, pretrained=True)
        model.to(cfg.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = get_scheduler(optimizer)
        criterion = nn.BCEWithLogitsLoss()
        metric_logger = MetricLogger(fold, group_name=cfg.RUN_NAME)
        # TRAIN LOOP
        best_score = 0.
        best_loss = np.inf
        for epoch in range(cfg.NUM_EPOCHS):
            start_time = time.time()
            train_avg_loss = train_one_epoch(model, optimizer, criterion, train_dataloader, metric_logger, cfg.DEVICE)
            val_avg_loss, preds = evaluate(model, criterion, val_dataloader, metric_logger, cfg.DEVICE)

            # Scheduler step
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_avg_loss)
            elif isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
            elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()

            score = roc_auc_score(val_labels, preds)
            metric_logger.avg_log(train_avg_loss, val_avg_loss, score)

            logger.info(f"Epoch:{epoch} | train_avg_loss:{train_avg_loss:.4f} | val_avg_loss:{val_avg_loss:.4f} | "
                        f"score: {score:.4f}")

            if score > best_score:
                best_score = score
                save_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL_TYPE}_fold_{fold}_best_score.pth.tar")
                save_checkpoint(save_path, model, optimizer, cfg.LEARNING_RATE, preds)
                logger.info(f"==> Found the best ROC_AUC score, save model to {save_path}")

            if val_avg_loss < best_loss:
                best_loss = val_avg_loss
                save_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL_TYPE}_fold_{fold}_best_val_loss.pth.tar")
                save_checkpoint(save_path, model, optimizer, cfg.LEARNING_RATE, preds)
                logger.info(f"==> Found the best validation loss, save model to {save_path}")
        # CV score of fold
        val_folds['preds'] = torch.load(os.path.join(cfg.OUTPUT_DIR,
                                                     f"{cfg.MODEL_TYPE}_fold_{fold}_best_val_loss.pth.tar"),
                                        map_location=torch.device("cpu"))['preds']
        oof_df = pd.concat([oof_df, val_folds])
        logger.info(f"========== Fold: {fold + 1} Result ==========")
        print_result(val_folds)
        metric_logger.finish()

        # NOTE remove after test code
        if args.test:
            break

    # CV score of all folds
    logger.info("==> Train done")
    logger.info("Cross validation score")
    print_result(oof_df)
    # Save result
    oof_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'oof_df.csv'), index=False)
    logger.info(f"Save cross validation score to {os.path.join(cfg.OUTPUT_DIR, 'oof_df.csv')}")

    if cfg.USE_GRAD_CAM:
        logger.info("==> GRAD CAM START")
        num_rows = 5
        for fold in range(cfg.NUM_FOLDS):
            model = get_model(cfg.EFF_VER, pretrained=False)
            state = torch.load(os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL_TYPE}_fold_{fold}_best_val_loss.pth.tar"),
                               map_location=torch.device("cpu"))['model']
            model.load_state_dict(state)
            model.to(cfg.DEVICE)
            model.eval()
            oof = pd.read_csv(os.path.join(cfg.OUTPUT_DIR, 'oof_df.csv'))
            oof = oof[oof['fold'] == fold].reset_index(drop=True)

            count = 0
            oof = oof.sort_values('preds', ascending=False)
            valid_dataset = GradCamGWDataset(oof)
            for i in range(len(valid_dataset)):
                image_id, series, vis_image, label = valid_dataset[i]
                result = get_grad_cam(model, cfg.DEVICE, series, vis_image, label)
                if result['vis'] is not None:
                    count += 1
                    metric_logger.fill_table(image_id, result['label'],  result['prob'], wandb.Image(result['img']),
                                             wandb.Image(result['vis']))
                if count >= num_rows:
                    break

            count = 0
            oof = oof.sort_values('preds', ascending=True)
            valid_dataset = GradCamGWDataset(oof)
            for i in range(len(valid_dataset)):
                image_id, series, vis_image, label = valid_dataset[i]
                result = get_grad_cam(model, cfg.DEVICE, series, vis_image, label)
                if result['vis'] is not None:
                    count += 1
                    metric_logger.fill_table(image_id, result['label'],  result['prob'], wandb.Image(result['img']),
                                             wandb.Image(result['vis']))
                if count >= num_rows:
                    break

        metric_logger.log_table()
        metric_logger.finish()
        logger.info("==> DONE")
