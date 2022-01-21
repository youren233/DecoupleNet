
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ['output', 'output', '../model', 'output']
_C.LOG_DIR = ['', '', '../log', '']
_C.DATA_DIR = ['', '', '../data', '']
_C.DESCRIPTION = ''
_C.ENV = 0
_C.EXP_ID = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.EPOCH_EVAL_FREQ = 10

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'

_C.MODEL.DECOUPLE = CN(new_allowed=True)
# Options: "" (no norm), "GN", "SyncBN".
# _C.MODEL.DECOUPLE.NORM = ""
# _C.MODEL.DECOUPLE.NUM_CONV = 0  # The number of convs in the mask head
# _C.MODEL.DECOUPLE.CONV_DIM = 256
# _C.MODEL.DECOUPLE.HEAD_CHANNELS = 64
# _C.MODEL.DECOUPLE.OCC_WEIGHT = 1
# _C.MODEL.DECOUPLE.OCCEE_WEIGHT = 1

_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ['model/hrnet_w32-36af842e.pth', 'model/hrnet_w32-36af842e.pth', '../model/hrnet_w32-36af842e.pth', 'model/hrnet_w32-36af842e.pth']
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.SE_MODULES = [False, False, True, True]

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.DATASET = 'mpii'
_C.DATASET.ROOT = ['/home/disk/weixing/datasets' ,'/home/bxx-wx/dataset', "", "F:\\ggy\\xx\\dataset"]

_C.DATASET.TRAIN_DATASET = 'mpii'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TRAIN_IMAGE_DIR = ''
_C.DATASET.TRAIN_ANNOTATION_FILE = 'train2017.json'
_C.DATASET.TRAIN_DATASET_TYPE = 'coco_lambda'

_C.DATASET.TEST_DATASET = 'mpii'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.TEST_IMAGE_DIR = ''
_C.DATASET.TEST_ANNOTATION_FILE = 'val2017.json'
_C.DATASET.TEST_DATASET_TYPE = 'coco_lambda'

_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False

_C.DATASET.SYNTHETIC_DATASET = 'synthetic'
_C.DATASET.SYNTHETIC_ROOT = ''

_C.DATASET.SYNTHETIC_TRAIN_DATASET = 'synthetic'
_C.DATASET.SYNTHETIC_TRAIN_SET = 'train'
_C.DATASET.SYNTHETIC_TRAIN_IMAGE_DIR = ''
_C.DATASET.SYNTHETIC_TRAIN_ANNOTATION_FILE = 'train2017.json'
_C.DATASET.SYNTHETIC_TRAIN_DATASET_TYPE = 'coco_lambda_syn'

_C.DATASET.SYNTHETIC_TEST_DATASET = 'synthetic'
_C.DATASET.SYNTHETIC_TEST_SET = 'valid'
_C.DATASET.SYNTHETIC_TEST_IMAGE_DIR = ''
_C.DATASET.SYNTHETIC_TEST_ANNOTATION_FILE = 'val2017.json'
_C.DATASET.SYNTHETIC_TEST_DATASET_TYPE = 'coco_lambda_syn'

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False

_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''
_C.TEST.BBOX_FRACTION = 1.0
_C.TEST.DECAY_THRE = 0.5
_C.TEST.SCALE_THRE = 1.25

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.local_rank:
        cfg.RANK = args.local_rank
    if args.exp_id:
        cfg.EXP_ID = args.exp_id
    cfg.LOG = cfg.RANK == 0
    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    env = cfg.ENV
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR[env]
    cfg.DATASET.ROOT = cfg.DATASET.ROOT[env]
    cfg.DATA_DIR = cfg.DATA_DIR[env]
    cfg.LOG_DIR = cfg.LOG_DIR[env]
    cfg.DATASET.ROOT = os.path.join(cfg.DATA_DIR, cfg.DATASET.ROOT)

    cfg.DATASET.TRAIN_IMAGE_DIR = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.TRAIN_IMAGE_DIR)
    cfg.DATASET.TRAIN_ANNOTATION_FILE = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.TRAIN_ANNOTATION_FILE)
    cfg.DATASET.TEST_IMAGE_DIR = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.TEST_IMAGE_DIR)
    cfg.DATASET.TEST_ANNOTATION_FILE = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.TEST_ANNOTATION_FILE)
    cfg.TEST.COCO_BBOX_FILE = os.path.join(cfg.DATASET.ROOT, cfg.TEST.COCO_BBOX_FILE)

    cfg.MODEL.PRETRAINED = cfg.MODEL.PRETRAINED[env]

    cfg.freeze()

    print("======> TRAIN_IMAGE_DIR: " + cfg.DATASET.TRAIN_IMAGE_DIR)
    print("======> TRAIN_ANNOTATION_FILE: " + cfg.DATASET.TRAIN_ANNOTATION_FILE)


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

