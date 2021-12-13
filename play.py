from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

DIR = os.path.abspath(__file__)  # 当前脚本
n = 3  # 与当前脚本的相对位置
for i in range(n):  # 第1次循环，得到父目录；的二次循环得到，父目录 的 父目录， 第3次得到 父目录 的 父目录 的父目录
    DIR = os.path.dirname(DIR)
sys.path.append(DIR)

import argparse
import os
import pprint
import shutil
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from lib.config import cfg
from lib.config import update_config
from lib.core.loss import JointsLambdaMSELoss
from lib.core.loss import JointsMSELoss
from lib.core.train import train_lambda
from lib.core.validate import validate_lambda
from lib.core.validate import validate_lambda_quantitative

# from utils.utils import get_last_layer_optimizer
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.utils import get_lambda_model_summary
from lib.utils.utils import set_seed

import lib.dataset
import lib.models

# --------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/crowdpose/hrnet/w32_256x192-xx.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0)
    parser.add_argument('--exp_id',
                        type=str,
                        default='exp_train')


    args = parser.parse_args()

    return args
# --------------------------------------------------------------------------------

def main():
    args = parse_args()
    update_config(cfg, args)
    model = eval('lib.models.'+"pose_hrnet_decouple"+'.get_pose_net')(
        cfg, is_train=True
    )

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )

    occ, occee = model(dump_input)
    print(occ.shape)
    print(occee.shape)
    print('over')


# dir = 'tools/lambda/output/PycharmTest_w32_256x192/lambda:0,1/val_results'
# file01 = os.path.join(dir, 'keypoints_test_results_epoch-1.json')
# file0 = os.path.join(dir, 'keypoints_test_results_mode0_epoch-1.json')
# file1 = os.path.join(dir, 'keypoints_test_results_mode1_epoch-1.json')
#
# test_result01 =  json.load(open(file01, 'r'))
# test_result0 =  json.load(open(file0, 'r'))
# test_result1 =  json.load(open(file1, 'r'))

# 网上：[[0, 2], [1, 3], [2, 4], [3, 5], [6, 8], [8, 10], [7, 9], [9, 11], [12, 13], [0, 13], [1, 13], [6,13],[7, 13]]
# 数据集：[[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11]]
# crowdpose = json.load(open("/home/disk/weixing/datasets/crowdpose/json/crowdpose_test.json", 'r'))

if __name__ == '__main__':
    main()