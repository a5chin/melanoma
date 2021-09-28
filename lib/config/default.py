from yacs.config import CfgNode as CN

_C = CN()

_C.DATASET = CN()
_C.DATASET.NAME = 'SIIM-ISIC Melanoma'
_C.DATASET.ROOT = r'\\aka\data\medical\siim-isic-melanoma-classification'
_C.DATASET.NUM_CLASSES = 2
_C.DATASET.IMAGE_TRAIN = _C.DATASET.ROOT + '/jpeg/train'
_C.DATASET.IMAGE_VAL = _C.DATASET.ROOT + '/jpeg/test'
_C.DATASET.IDS_TRAIN = _C.DATASET.ROOT + '/train.csv'
_C.DATASET.IDS_VAL = _C.DATASET.ROOT + '/test.csv'
_C.DATASET.NUM_CLASSES_AGE = 17
_C.DATASET.NUM_CLASSES_SITE = 6
_C.DATASET.BATCH_SIZE = 128
_C.DATASET.TOTAL_EPOCH = 20

_C.MODEL = CN()
_C.MODEL.NAME = ''
_C.MODEL.LOG_DIR = './logs'
_C.MODEL.SAVED_DIR = './saved_models'
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
