import logging
from easydict import EasyDict as edict

__C = edict()
# for consumers
cfg = __C
# Init logger
logger = logging.getLogger()
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)

# CONFIG

# NAMES
__C.PROJECT_NAME = "Gravitation wave detection"
__C.RUN_NAME = "GWD_BASELINE"
__C.MODEL_TYPE = None
# ----------------------------------------------------
# GLOBAL
__C.USE_GRAD_CAM = False
__C.USE_APEX = True
__C.EFF_VER = 3
__C.DEVICE = None
__C.IMG_CHANNELS = 1
__C.NUM_CLASSES = 1
__C.NUM_EPOCHS = 1
__C.BATCH_SIZE = 48
__C.LEARNING_RATE = 1e-4
__C.NUM_WORKERS = 2
__C.SEED = 1446
__C.Q_TRANSFORM_PARAMS = {"sr": 2048, "fmin": 20, "fmax": 1024, "hop_length": 32, "bins_per_octave": 8}
# OPTIMIZER ADAM
__C.WEIGHT_DECAY = 1e-6
__C.BETAS = (0.0, 0.99)
# ----------------------------------------------------
# SCHEDULER
# [ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts]
__C.SCHEDULER_VERSION = 'CosineAnnealingWarmRestarts'
__C.FACTOR = 0.2
__C.PATIENCE = 4
__C.EPS = 1e-6
__C.T_MAX = 6
__C.MIN_LR = 1e-6
__C.T_0 = 10     # scheduler restarts after Ti epochs.
# -----------------------------------------------------
# CROSS VALIDATION
__C.NUM_FOLDS = 5
__C.FOLD_LIST = [i for i in range(__C.NUM_FOLDS)]
# METRICS & WANDB
__C.LOSS_FREQ = 100
__C.WANDB_ID = None
__C.RESUME_ID = None
# PATHS
__C.DATA_FOLDER = None
__C.OUTPUT_DIR = './'
__C.MODEL_DIR = None
