import argparse
from config import cfg
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

    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    seed_everything(cfg.SEED)
    args = parse_args()

    assert args.data_path, 'Data path not specified'
    assert args.model_dir, 'Model path not specified'
    assert args.device in ['gpu', 'cpu'], 'incorrect device type must be gpu or cpu'
    assert args.eff_ver in [0, 1, 2, 3, 4, 5, 6, 7], 'Efficient version must be int and in range 0-7'

    if args.device == 'gpu':
        cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cpu':
        cfg.DEVICE = 'cpu'

    if args.eff_ver:
        cfg.EFF_VER = args.eff_ver
        cfg.MODEL_TYPE = f"tf_efficientnet_b{cfg.EFF_VER}_ns"

