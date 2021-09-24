import argparse
import time
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from config import cfg
from tqdm import tqdm
from dataset import GWDataset
from models import get_model
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Gravitation wave detection')
    parser.add_argument('--data_path', dest='data_path', help='Path to root dataset', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--model_dir', dest='model_dir', help='Path where models stores', default=None, type=str)
    parser.add_argument('--device', dest='device', help='Use device: gpu or cpu. Default use gpu if available',
                        default='gpu', type=str)
    parser.add_argument('--eff_ver', dest='eff_ver', help='Efficient version', default=4, type=int)
    parser.add_argument('--oof', dest='oof', help='Path to oof score', default=None, type=str)
    parser.add_argument('--ckpt', dest='ckpt', help='Path to model ckpt.pth.tar', default=None, type=str)
    parser.add_argument('--run_name', dest='run_name', help='Run name of wandb', default=None, type=str)
    parser.add_argument('--model_name', dest='model_name', help='Model name', default=None, type=str)

    parser.print_help()
    return parser.parse_args()


def inference(model, states, dataloader):
    loop = tqdm(dataloader, leave=True)
    preds = []
    for batch_idx, (img, _) in enumerate(loop):
        img = img.to(cfg.DEVICE)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                y_preds = model(img)
            avg_preds.append(y_preds.sigmoid().cpu().numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        preds.append(avg_preds)
    preds = np.concatenate(preds)
    return preds


if __name__ == '__main__':
    seed_everything(cfg.SEED)
    args = parse_args()

    logger = logging.getLogger('inference')

    assert args.data_path, 'Data path not specified'
    assert args.model_dir, 'Model path not specified'
    assert args.device in ['gpu', 'cpu'], 'incorrect device type must be gpu or cpu'
    assert args.eff_ver in [0, 1, 2, 3, 4, 5, 6, 7], 'Efficient version must be int and in range 0-7'

    cfg.DATA_FOLDER = args.data_path

    if args.device == 'gpu':
        cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cpu':
        cfg.DEVICE = 'cpu'

    if args.eff_ver:
        cfg.EFF_VER = args.eff_ver
        cfg.MODEL_TYPE = None
        cfg.NUM_FOLDS = 1

    if args.model_name:
        cfg.MODEL_TYPE = args.model_name

    logger.info(f'==> Start {__name__} at {time.ctime()}')
    logger.info(f'==> Config params: {cfg.__dict__}')
    logger.info(f'==> Called with args: {args.__dict__}')
    logger.info(f'==> Using device:{args.device}')

    if args.oof:
        oof = pd.read_csv(args.oof)
        logger.info('==> Loaded cross validation score')
        print_result(oof)

    sub_path = os.path.join(cfg.DATA_FOLDER, 'sample_submission.csv')
    test_df = pd.read_csv(sub_path)
    test_df['img_path'] = test_df['id'].apply(get_test_file_path)

    model = get_model(cfg.EFF_VER, pretrained=False).to(cfg.DEVICE)
    states = [torch.load(os.path.join(args.model_dir, f"{cfg.MODEL_TYPE}_fold_{fold}_best_val_loss.pth.tar"))
              for fold in cfg.FOLD_LIST]
    test_dataset = GWDataset(test_df, use_filter=True, use_transform=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, num_workers=2, pin_memory=True)

    predictions = inference(model, states, test_dataloader)

    test_df['target'] = predictions
    test_df[['id', 'target']].to_csv('submission.csv', index=False)
    logger.info(f"==> Test done. Save submission")