from utils.ordered_easydict import OrderedEasyDict as edict
import os

__C = edict()
cfg = __C
__C.GLOBAL = edict()
__C.GLOBAL.AUGMENT = True
__C.GLOBAL.BATCH_SZIE = 8
__C.GLOBAL.SIZE = 320
__C.GLOBAL.EPOCHS = 20
__C.GLOBAL.LR = 1e-4
__C.GLOBAL.MOMENTUM = 0.9
__C.GLOBAL.WEIGHT_DECAY = 1e-4
__C.GLOBAL.POWER = 0.9
__C.GLOBAL.MODEL_SAVE_DIR = None
__C.GLOBAL.LOSS = 'CE'
__C.GLOBAL.TRAIN_RULE = 'None'
__C.GLOBAL.MODEL_SAVE_DIR = '/home/jinqizhao2/PycharmProjects/Sequence_Precipitation/experiments/MULTIDATA' \
                            + '_' + str(__C.GLOBAL.LR) + '_' + __C.GLOBAL.TRAIN_RULE + '_' + __C.GLOBAL.LOSS
assert __C.GLOBAL.MODEL_SAVE_DIR is not None

__C.LAPS = edict()
__C.LAPS.NUM_CLASS = 5
__C.LAPS.IN_LEN = 1
__C.LAPS.OUT_LEN = 1
__C.LAPS.STRIDE = 3
__C.LAPS.WEIGHT = [1., 7., 33., 126., 300.]
__C.LAPS.CLS_NUM = [933167353, 128541737, 28198230, 7352825, 3102391]
__C.LAPS.CLS_RATIO = [0.8480544570266976, 0.11681762400533055, 0.025626308673235308, 0.006682184061562779, 0.0028194262331737545]


__C.MODEL = edict()
# __C.MODEL.NAME = 'ConvGRU'
from models.Encoder_Forecaster import activation
__C.MODEL.RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)
__C.MODEL.NAME = 'STIN'

