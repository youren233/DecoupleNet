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

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import JointsLambdaMSELoss
from lib.core.loss import JointsMSELoss
from lib.core.validate import validate_dcp_gcn
from lib.utils.utils import create_logger
from lib.utils.utils import get_dcp_model_summary
from lib.utils.utils import set_seed

import lib.dataset
import lib.models

# --------------------------------------------------------------------------------
set_seed(seed_id=0)

# --------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/crowdpose/hrnet/w32_256x192-decouple-gcn.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

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
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--exp_id',
                        type=str,
                        default='Train_Dcp')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    device = cfg.GPUS[args.local_rank]
    torch.cuda.set_device(device)

    model = eval('lib.models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (16, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )

    logger.info(get_dcp_model_summary(model, dump_input))

    if cfg.TEST.MODEL_FILE:
        model_file = os.path.join(final_output_dir, cfg.TEST.MODEL_FILE)
        logger.info('=> loading model from {}'.format(model_file))
        model_object = torch.load(model_file, map_location='cpu')
        if 'latest_state_dict' in model_object.keys():
            logger.info('=> loading from latest_state_dict at {}'.format(model_file))
            model.load_state_dict(model_object['latest_state_dict'], strict=False)
        elif 'state_dict' in model_object.keys():
            logger.info('=> loading from latest_state_dict at {}'.format(model_file))
            model.load_state_dict(model_object['state_dict'], strict=False)
        else:
            logger.info('=> no latest_state_dict found')
            model.load_state_dict(model_object, strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file, map_location='cpu'))

    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    
    # ------------------------------------------------
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('lib.dataset.'+cfg.DATASET.TEST_DATASET)(
        cfg=cfg, image_dir=cfg.DATASET.TEST_IMAGE_DIR, annotation_file=cfg.DATASET.TEST_ANNOTATION_FILE,
        dataset_type=cfg.DATASET.TEST_DATASET_TYPE,
        image_set=cfg.DATASET.TEST_SET, is_train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # # # evaluate on validation set
    validate_dcp_gcn(cfg, valid_loader, valid_dataset, model,
                     os.path.join(final_output_dir, 'decouple_gcn'), writer_dict, lambda_vals=[0, 1], log=logger)


    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
