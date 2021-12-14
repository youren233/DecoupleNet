# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

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

import _init_paths
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import JointsMSELoss
from lib.core.train import train_dcp
from lib.core.validate import validate_dcp

import lib.dataset
import lib.models
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.utils import get_dcp_model_summary
from lib.utils.utils import set_seed


# --------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/crowdpose/hrnet/w32_256x192-decouple.yaml',
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
                        default='Train_Dcp')


    args = parser.parse_args()

    return args
# --------------------------------------------------------------------------------

def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    dist = False
    # if len(cfg.GPUS) > 1:
    #     dist = True

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # 分布式训练 1）
    if dist:
        torch.distributed.init_process_group('nccl', init_method='env://')
    set_seed(seed_id=0)

    model = eval('lib.models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=True)

    writer_dict = None
    if cfg.LOG:
        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    dump_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    ### this is used to visualize the network
    ### throws an assertion error on cube3, works well on bheem
    ### commented for now
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_dcp_model_summary(model, dump_input))

    if cfg.ENV != 2:
        device = cfg.GPUS[args.local_rank]
        torch.cuda.set_device(device)
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)

    if dist:
        model = torch.nn.parallel.DistributedDataParallel(model)

     # ------------------------------------------
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = eval('lib.dataset.'+cfg.DATASET.TRAIN_DATASET)(
        cfg=cfg, image_dir=cfg.DATASET.TRAIN_IMAGE_DIR, annotation_file=cfg.DATASET.TRAIN_ANNOTATION_FILE,
        dataset_type=cfg.DATASET.TRAIN_DATASET_TYPE,
        image_set=cfg.DATASET.TRAIN_SET, is_train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
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
    
    train_sampler = None
    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler
    )

    val_sampler = None
    if dist:
        val_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=val_sampler
    )

    # # # # # ---------------------------------------------
    best_perf = 0.0
    perf_indicator = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint_resume.pth'
    )
    # # # # ----------------------------------------------
    logger.info('=> updated lr schedule is {}'.format(cfg.TRAIN.LR_STEP))
    # 断点自动恢复训练
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("[AUTO_RESUME] ======> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = begin_epoch - 1
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )
    # ----------------------------------------------------
    criterion = criterion.cuda()
    model = model.cuda()
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        # # # train for one epoch
        if cfg.LOG:
            logger.info('====== training on lambda, lr={}, {} th epoch ======'
                        .format(optimizer.state_dict()['param_groups'][0]['lr'], epoch))
        # train_dcp(cfg, train_loader, model, criterion, optimizer, epoch,
        #   final_output_dir, tb_log_dir, writer_dict, print_prefix='lambda')

        lr_scheduler.step()

        if epoch % cfg.EPOCH_EVAL_FREQ == 0 or epoch > 205:
            perf_indicator = validate_dcp(cfg, valid_loader, valid_dataset, model, criterion,
                     final_output_dir, tb_log_dir, writer_dict, epoch=epoch, print_prefix='lambda', lambda_vals=[0, 1], log=logger)

            if perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            if cfg.LOG:
                logger.info('=> model AP: {} | saving checkpoint to {}'.format(perf_indicator, final_output_dir))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': cfg.MODEL.NAME,
                    'state_dict': model.state_dict(),
                    'latest_state_dict': model.module.state_dict(),
                    'best_state_dict': model.module.state_dict(),
                    'perf': perf_indicator,
                    'optimizer': optimizer.state_dict(),
                    }, best_model, final_output_dir, filename='checkpoint_{}.pth'.format(epoch + 1))

    # # ----------------------------------------------
    if cfg.LOG:
        final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
        torch.save(model.module.state_dict(), final_model_state_file)
        writer_dict['writer'].close()

# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()