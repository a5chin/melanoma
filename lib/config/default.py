from yacs.config import CfgNode as CN
from pathlib import Path

_C = CN()

_C.DATASET = CN()
_C.DATASET.NAME = 'SIIM-ISIC Melanoma'
_C.DATASET.ROOT = 'path to root'
_C.DATASET.NUM_CLASSES = 2
_C.DATASET.IMAGE_TRAIN = _C.DATASET.ROOT / Path('jpeg/train')
_C.DATASET.IMAGE_VAL = _C.DATASET.ROOT / Path('jpeg/test')
_C.DATASET.IDS_TRAIN = _C.DATASET.ROOT / Path('train.csv')
_C.DATASET.IDS_VAL = _C.DATASET.ROOT / Path('test.csv')
_C.DATASET.NUM_CLASSES_AGE = 17
_C.DATASET.NUM_CLASSES_SITE = 6
_C.DATASET.BATCH_SIZE = 128
_C.DATASET.TOTAL_EPOCH = 20

_C.MODEL = CN()
_C.MODEL.NAME = ''
_C.MODEL.LOG_DIR = Path('./logs')
_C.MODEL.SAVED_DIR = Path('./saved_models')
_C.MODEL.PRETRAINED = ''
_C.MODEL.OPTIMIZER = ''
_C.MODEL.CRITERION = ''


def get_defaults():
    return _C.clone()


def load_config(config_path):
    cfg = get_defaults()
    cfg.merge_from_file(config_path)
    cfg.freeze()
    return cfg
