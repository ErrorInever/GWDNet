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

# GLOBAL
__C.DEVICE = None
__C.IMG_CHANNELS = 1
__C.NUM_CLASSES = 1
__C.SEED = 1446