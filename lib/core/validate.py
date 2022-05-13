# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from lib.core.evaluate import accuracy
from lib.core.inference import get_final_preds
from lib.utils.transforms import flip_back
from lib.utils.vis import save_debug_images
from tqdm import tqdm

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, epoch=-1, print_prefix=''):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6+1)) ## update to add annotation ids
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            annotation_id = meta['annotation_id'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1) ## this is the area of the detection
            all_boxes[idx:idx + num_images, 5] = score
            all_boxes[idx:idx + num_images, 6] = annotation_id
            image_path.extend(meta['image'])

            idx += num_images

            if (i % config.PRINT_FREQ == 0) or (i == (len(val_loader)-1)):
                save_size = 16
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.6f} ({loss.avg:.6f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader)-1, batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'val'), epoch, i, print_prefix)

                save_debug_images(config, input[:save_size, [2,1,0], :, :], meta, target[:save_size], (pred*4)[:save_size], output[:save_size],
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path, epoch,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# --------------------------------------------------------------------------------
def validate_lambda_quantitative(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, epoch=-1, print_prefix='', lambda_vals=[0, 1], log=logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # switch to evaluate mode
    model.eval()

    num_samples = len(lambda_vals)*len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6+1+1)) ## update to add annotation ids and mode l=0,1
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        val_loader = tqdm(val_loader)
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            B, C, H, W = input.shape

            for lambda_idx, lambda_val in enumerate(lambda_vals):
                # b x [0, 1], b x [1, 0]
                lambda_a = lambda_val*torch.ones(B, 1).cuda()
                lambda_vec = torch.cat([lambda_a, 1 - lambda_a], dim=1)

                input = input.cuda()
                outputs = model(input, lambda_vec)
                output = outputs

                if config.TEST.FLIP_TEST:
                    input_flipped = input.flip(3)
                    outputs_flipped = model(input_flipped, lambda_vec)
                    output_flipped = outputs_flipped

                    output_flipped = flip_back(output_flipped.cpu().numpy(),
                                               val_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                loss = criterion(output, target, target_weight)

                num_images = input.size(0)
                # measure accuracy and record loss
                losses.update(loss.item(), num_images)
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                 target.cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                
                if lambda_val == 0:
                    score = meta['score'].numpy()*config.TEST.DECAY_THRE

                else:
                    score = meta['score'].numpy()

                annotation_id = meta['annotation_id'].numpy()

                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                all_boxes[idx:idx + num_images, 6] = annotation_id
                all_boxes[idx:idx + num_images, 7] = lambda_a.detach().cpu().numpy().reshape(-1)

                image_path.extend(meta['image'])

                idx += num_images

                if config.LOG:
                    msg = 'Loss {loss.val:.6f} ' \
                          'Accuracy {acc.val:.3f} ' \
                          'Lambda_a {lambda_val:.3f}'.format(
                        loss=losses, acc=acc, lambda_val=lambda_val)
                    val_loader.set_description(msg)

                # if ((i % config.PRINT_FREQ == 0) or (i == (len(val_loader)-1))) and config.LOG:
                #     save_size = min(16, B)
                    # meta['pred_joints_vis'] = torch.ones_like(meta['joints_vis'])
                    
                    # prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'val'), epoch, i, print_prefix)
                    # suffix = 'a'
                    # for count in range(min(save_size, len(lambda_a))):
                    #     suffix += '_[{}:{}]'.format(count, round(lambda_a[count].item(), 2))

                    # save_debug_images(config, input[:save_size, [2,1,0], :, :], meta, target[:save_size], (pred*4)[:save_size], output[:save_size],
                    #                   prefix, suffix)

        perf_indicator = 0.0
        if config.LOG:
            name_values, name_values_mode0, \
            name_values_mode1, name_values_mode2, \
            name_values_mode3, perf_indicator = val_dataset.evaluate(
                                        config, all_preds, output_dir, all_boxes, image_path, epoch,
                                        filenames, imgnums
                                    )

            model_name = config.MODEL.NAME

            _print_name_value(name_values, 'total:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode0, 'occer:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode1, 'occed:{}'.format(model_name), log=log)

        if writer_dict and config.LOG:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    val_loader.close()
    return perf_indicator

# dcp
def validate_dcp_gcn(config, val_loader, val_dataset, model, output_dir,
                     writer_dict=None, epoch=-1, lambda_vals=[0, 1], log=logger):
    accAMer = AverageMeter()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # switch to evaluate mode
    model.eval()

    num_samples = len(lambda_vals)*len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6+1+1)) ## update to add annotation ids and mode l=0,1
    image_path = []
    filenames = []
    imgnums = []
    global_idx = 0
    with torch.no_grad():
        val_loader = tqdm(val_loader)
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            B, C, H, W = input.shape
            # get output
            input = input.cuda()
            outputs = model(input)
            occ_pose, occee_pose = outputs
            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)
                occ_pose_flipped, occee_pose_flipped = outputs_flipped
                # occ
                occ_pose_flipped = flip_back(occ_pose_flipped.cpu().numpy(), val_dataset.flip_pairs)
                occ_pose_flipped = torch.from_numpy(occ_pose_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    occ_pose_flipped[:, :, :, 1:] = occ_pose_flipped.clone()[:, :, :, 0:-1]
                occ_pose = (occ_pose + occ_pose_flipped) * 0.5

                # occee
                occee_pose_flipped = flip_back(occee_pose_flipped.cpu().numpy(), val_dataset.flip_pairs)
                occee_pose_flipped = torch.from_numpy(occee_pose_flipped.copy()).cuda()

                if config.TEST.SHIFT_HEATMAP:
                    occee_pose_flipped[:, :, :, 1:] = occee_pose_flipped.clone()[:, :, :, 0:-1]

                occee_pose = (occee_pose + occee_pose_flipped) * 0.5
            for m, output in enumerate([occ_pose, occee_pose]):
                # b x [0, 1], b x [1, 0]
                mode = m * torch.ones(B, 1)

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                num_images = input.size(0)
                # measure accuracy and record loss
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                 target.cpu().numpy())
                accAMer.update(avg_acc, cnt)

                # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                # todo 被遮挡者 score * 0.5？
                if m == 1:
                    score = meta['score'].numpy()*config.TEST.DECAY_THRE
                else:
                    score = meta['score'].numpy()

                annotation_id = meta['annotation_id'].numpy()

                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

                all_preds[global_idx:global_idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[global_idx:global_idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[global_idx:global_idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[global_idx:global_idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[global_idx:global_idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[global_idx:global_idx + num_images, 5] = score
                all_boxes[global_idx:global_idx + num_images, 6] = annotation_id
                all_boxes[global_idx:global_idx + num_images, 7] = mode.detach().cpu().numpy().reshape(-1)

                image_path.extend(meta['image'])

                global_idx += num_images

                if config.LOG:
                    msg = 'Accuracy {acc.val:.8f}'.format(acc=accAMer)
                    val_loader.set_description(msg)

                # 可视化结果
                if i % config.PRINT_FREQ == 0:
                    save_size = min(2, B)
                    meta['pred_joints_vis'] = torch.ones_like(meta['joints_vis'])

                    suffix = str(i) + ['ru', 'rd'][m]

                    save_debug_images(config, input[:save_size, [2,1,0], :, :], meta, target[:save_size], (pred*4)[:save_size], output[:save_size],
                                      output_dir, suffix)

        perf_indicator = 0.0
        if config.LOG:
            name_values, name_values_mode0, \
            name_values_mode1, name_values_mode2, \
            name_values_mode3, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path, epoch,
                filenames, imgnums
            )

            model_name = config.MODEL.NAME

            _print_name_value(name_values, 'total:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode0, 'occer:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode1, 'occed:{}'.format(model_name), log=log)

        if writer_dict and config.LOG:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_acc',
                accAMer.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    val_loader.close()
    return perf_indicator
# dcp

# dcp-cnn
def validate_dcp_cnn(config, val_loader, val_dataset, model, output_dir, writer_dict=None,
                     epoch=-1, lambda_vals=[0, 1], log=logger):
    accAMer = AverageMeter()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # switch to evaluate mode
    model.eval()

    num_outputs = len(lambda_vals)*len(val_dataset)
    all_preds = np.zeros(
        (num_outputs, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_outputs, 6+1+1)) ## update to add annotation ids and mode l=0,1
    image_path = []
    filenames = []
    imgnums = []
    global_idx = 0
    with torch.no_grad():
        val_loader = tqdm(val_loader)
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            B, C, H, W = input.shape
            # get output
            input = input.cuda()
            pose_dict = model(input)
            ru = pose_dict['ru']
            rd = pose_dict['rd']

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                pose_dict_flipped = model(input_flipped)
                ru_flipped = pose_dict_flipped['ru']
                rd_flipped = pose_dict_flipped['rd']
                # occ
                ru_flipped = flip_back(ru_flipped.cpu().numpy(), val_dataset.flip_pairs)
                ru_flipped = torch.from_numpy(ru_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    ru_flipped[:, :, :, 1:] = ru_flipped.clone()[:, :, :, 0:-1]
                ru = (ru + ru_flipped) * 0.5

                # occee
                rd_flipped = flip_back(rd_flipped.cpu().numpy(), val_dataset.flip_pairs)
                rd_flipped = torch.from_numpy(rd_flipped.copy()).cuda()

                if config.TEST.SHIFT_HEATMAP:
                    rd_flipped[:, :, :, 1:] = rd_flipped.clone()[:, :, :, 0:-1]

                rd = (rd + rd_flipped) * 0.5
            for m, output in enumerate([ru, rd]):
                # b x [0, 1], b x [1, 0]
                mode = m * torch.ones(B, 1)

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                num_images = input.size(0)
                # measure accuracy and record loss
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                 target.cpu().numpy())
                accAMer.update(avg_acc, cnt)

                # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                # todo 被遮挡者结果权重衰减？
                if m == 1:
                    score = meta['score'].numpy()*config.TEST.DECAY_THRE
                else:
                    score = meta['score'].numpy()

                annotation_id = meta['annotation_id'].numpy()

                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

                all_preds[global_idx:global_idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[global_idx:global_idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[global_idx:global_idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[global_idx:global_idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[global_idx:global_idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[global_idx:global_idx + num_images, 5] = score
                all_boxes[global_idx:global_idx + num_images, 6] = annotation_id
                all_boxes[global_idx:global_idx + num_images, 7] = mode.detach().cpu().numpy().reshape(-1)

                image_path.extend(meta['image'])

                global_idx += num_images

                if config.LOG:
                    msg = 'Accuracy {acc.val:.8f}'.format(acc=accAMer)
                    val_loader.set_description(msg)

                # 可视化结果
                if i % config.PRINT_FREQ == 0:
                    save_size = min(8, B)
                    meta['pred_joints_vis'] = torch.ones_like(meta['joints_vis'])

                    suffix = str(epoch) + '_' + str(i) + ['_ru', '_rd'][m]

                    save_debug_images(config, input[:save_size, [2,1,0], :, :], meta, target[:save_size], (pred*4)[:save_size], output[:save_size],
                                      output_dir, suffix)
        perf_indicator = 0.0
        if config.LOG:
            name_values, name_values_mode0, \
            name_values_mode1, name_values_mode2, \
            name_values_mode3, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path, epoch,
                filenames, imgnums
            )

            model_name = config.MODEL.NAME

            _print_name_value(name_values, 'total:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode0, 'occer:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode1, 'occed:{}'.format(model_name), log=log)

        if writer_dict and config.LOG:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_acc',
                accAMer.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    val_loader.close()
    return perf_indicator
# dcp-cnn

# one
def validate_one(config, val_loader, val_dataset, model, output_dir, writer_dict=None,
                     epoch=-1, log=logger):
    accAMer = AverageMeter()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # switch to evaluate mode
    model.eval()

    num_outputs = len(val_dataset)
    all_preds = np.zeros(
        (num_outputs, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_outputs, 6+1)) ## update to add annotation ids
    image_path = []
    filenames = []
    imgnums = []
    global_idx = 0
    with torch.no_grad():
        val_loader = tqdm(val_loader)
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            B, C, H, W = input.shape
            # get output
            input = input.cuda()
            pose = model(input)

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                flipped_pose = model(input_flipped)
                # occ
                flipped_pose = flip_back(flipped_pose.cpu().numpy(), val_dataset.flip_pairs)
                flipped_pose = torch.from_numpy(flipped_pose.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    flipped_pose[:, :, :, 1:] = flipped_pose.clone()[:, :, :, 0:-1]
                pose = (pose + flipped_pose) * 0.5

            # b x [0, 1], b x [1, 0]
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            num_images = input.size(0)
            # measure accuracy and record loss
            _, avg_acc, cnt, pred = accuracy(pose.cpu().numpy(),
                                             target.cpu().numpy())
            accAMer.update(avg_acc, cnt)

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            # todo 被遮挡者结果权重衰减？
            score = meta['score'].numpy()

            annotation_id = meta['annotation_id'].numpy()

            preds, maxvals = get_final_preds(
                config, pose.clone().cpu().numpy(), c, s)

            all_preds[global_idx:global_idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[global_idx:global_idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[global_idx:global_idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[global_idx:global_idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[global_idx:global_idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[global_idx:global_idx + num_images, 5] = score
            all_boxes[global_idx:global_idx + num_images, 6] = annotation_id

            image_path.extend(meta['image'])

            global_idx += num_images

            if config.LOG:
                msg = 'Accuracy {acc.val:.8f}'.format(acc=accAMer)
                val_loader.set_description(msg)

            # 可视化结果
            # if i % config.PRINT_FREQ == 0:
            #     save_size = min(8, B)
            #     meta['pred_joints_vis'] = torch.ones_like(meta['joints_vis'])
            #
            #     # suffix = str(epoch) + '_' + str(i) + ['_ru', '_rd'][m]
            #
            #     save_debug_images(config, input[:save_size, [2,1,0], :, :], meta, target[:save_size], (pred*4)[:save_size], output[:save_size],
            #                           output_dir, suffix)
        perf_indicator = 0.0
        if config.LOG:
            name_values, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path, epoch,
                filenames, imgnums
            )

            model_name = config.MODEL.NAME

            _print_name_value(name_values, 'total:{}'.format(model_name), log=log)

        if writer_dict and config.LOG:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_acc',
                accAMer.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    val_loader.close()
    return perf_indicator
# one

# dcp-cr
def validate_dcp_cr(config, val_loader, val_dataset, model, output_dir, writer_dict=None,
                     epoch=-1, lambda_vals=[0, 1], log=logger):
    accAMer = AverageMeter()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # switch to evaluate mode
    model.eval()

    num_outputs = len(lambda_vals)*len(val_dataset)
    all_preds = np.zeros(
        (num_outputs, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_outputs, 6+1+1)) ## update to add annotation ids and mode l=0,1
    image_path = []
    filenames = []
    imgnums = []
    global_idx = 0
    with torch.no_grad():
        val_loader = tqdm(val_loader)
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            B, C, H, W = input.shape
            # get output
            input = input.cuda()
            pose_dict = model(input)
            cu = pose_dict['cu']
            cd = pose_dict['cd']
            ru = pose_dict['ru']
            rd = pose_dict['rd']

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                pose_dict_flipped = model(input_flipped)
                cu_flipped = pose_dict_flipped['cu']
                cd_flipped = pose_dict_flipped['cd']
                ru_flipped = pose_dict_flipped['ru']
                rd_flipped = pose_dict_flipped['rd']
                # occ
                ru_flipped = flip_back(ru_flipped.cpu().numpy(), val_dataset.flip_pairs)
                ru_flipped = torch.from_numpy(ru_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    ru_flipped[:, :, :, 1:] = ru_flipped.clone()[:, :, :, 0:-1]
                ru = (ru + ru_flipped) * 0.5

                # occee
                rd_flipped = flip_back(rd_flipped.cpu().numpy(), val_dataset.flip_pairs)
                rd_flipped = torch.from_numpy(rd_flipped.copy()).cuda()

                if config.TEST.SHIFT_HEATMAP:
                    rd_flipped[:, :, :, 1:] = rd_flipped.clone()[:, :, :, 0:-1]

                rd = (rd + rd_flipped) * 0.5

                # ----------------- c ---------------
                # occ
                # cu_flipped = flip_back(cu_flipped.cpu().numpy(), val_dataset.flip_pairs)
                # cu_flipped = torch.from_numpy(cu_flipped.copy()).cuda()
                #
                # # feature is not aligned, shift flipped heatmap for higher accuracy
                # if config.TEST.SHIFT_HEATMAP:
                #     cu_flipped[:, :, :, 1:] = cu_flipped.clone()[:, :, :, 0:-1]
                # cu = (cu + cu_flipped) * 0.5
                #
                # # occee
                # cd_flipped = flip_back(cd_flipped.cpu().numpy(), val_dataset.flip_pairs)
                # cd_flipped = torch.from_numpy(cd_flipped.copy()).cuda()
                #
                # if config.TEST.SHIFT_HEATMAP:
                #     cd_flipped[:, :, :, 1:] = cd_flipped.clone()[:, :, :, 0:-1]
                #
                # cd = (cd + cd_flipped) * 0.5
            for m, output in enumerate([ru, rd]):
                # b x [0, 1], b x [1, 0]
                mode = m * torch.ones(B, 1)

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                num_images = input.size(0)
                # measure accuracy and record loss
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                 target.cpu().numpy())
                accAMer.update(avg_acc, cnt)

                # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                # todo 被遮挡者结果权重衰减？
                if m == 1 or m == 3:
                    score = meta['score'].numpy()*config.TEST.DECAY_THRE
                else:
                    score = meta['score'].numpy()

                annotation_id = meta['annotation_id'].numpy()

                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

                all_preds[global_idx:global_idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[global_idx:global_idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[global_idx:global_idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[global_idx:global_idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[global_idx:global_idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[global_idx:global_idx + num_images, 5] = score
                all_boxes[global_idx:global_idx + num_images, 6] = annotation_id
                all_boxes[global_idx:global_idx + num_images, 7] = mode.detach().cpu().numpy().reshape(-1)

                image_path.extend(meta['image'])

                global_idx += num_images

                if config.LOG:
                    msg = 'Accuracy {acc.val:.8f}'.format(acc=accAMer)
                    val_loader.set_description(msg)

                # 可视化结果
                # if i % config.PRINT_FREQ == 0:
                #     save_size = min(8, B)
                #     meta['pred_joints_vis'] = torch.ones_like(meta['joints_vis'])
                #
                #     suffix = str(epoch) + '_' + str(i) + ['_ru', '_rd'][m]
                #
                #     save_debug_images(config, input[:save_size, [2,1,0], :, :], meta, target[:save_size], (pred*4)[:save_size], output[:save_size],
                #                       output_dir, suffix)
        perf_indicator = 0.0
        if config.LOG:
            name_values, name_values_mode0, \
            name_values_mode1, name_values_mode2, \
            name_values_mode3, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path, epoch,
                filenames, imgnums
            )

            model_name = config.MODEL.NAME

            _print_name_value(name_values, 'total:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode0, 'up:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode1, 'down:{}'.format(model_name), log=log)
            # _print_name_value(name_values_mode2, 'ru:{}'.format(model_name), log=log)
            # _print_name_value(name_values_mode3, 'rd:{}'.format(model_name), log=log)

        if writer_dict and config.LOG:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_acc',
                accAMer.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    val_loader.close()
    return perf_indicator
# dcp-cr

# dcp-naive
def validate_dcp_naive(config, val_loader, val_dataset, model, output_dir, writer_dict=None,
                     epoch=-1, lambda_vals=[0, 1], log=logger):
    accAMer = AverageMeter()

    visualize_dir = os.path.join(output_dir, "visualize")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)

    # switch to evaluate mode
    model.eval()

    num_outputs = len(lambda_vals)*len(val_dataset)
    all_preds = np.zeros(
        (num_outputs, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_outputs, 6+1+1)) ## update to add annotation ids and mode l=0,1
    image_path = []
    filenames = []
    imgnums = []
    global_idx = 0
    with torch.no_grad():
        val_loader = tqdm(val_loader)
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            B, C, H, W = input.shape
            # get output
            input = input.cuda()
            pose_dict = model(input)
            up_pose = pose_dict['up']
            down_pose = pose_dict['down']

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                pose_dict_flipped = model(input_flipped)
                up_flipped = pose_dict_flipped['up']
                down_flipped = pose_dict_flipped['down']
                # occ
                up_flipped = flip_back(up_flipped.cpu().numpy(), val_dataset.flip_pairs)
                up_flipped = torch.from_numpy(up_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    up_flipped[:, :, :, 1:] = up_flipped.clone()[:, :, :, 0:-1]
                up_pose = (up_pose + up_flipped) * 0.5

                # occee
                down_flipped = flip_back(down_flipped.cpu().numpy(), val_dataset.flip_pairs)
                down_flipped = torch.from_numpy(down_flipped.copy()).cuda()

                if config.TEST.SHIFT_HEATMAP:
                    down_flipped[:, :, :, 1:] = down_flipped.clone()[:, :, :, 0:-1]

                down_pose = (down_pose + down_flipped) * 0.5
            for m, output in enumerate([up_pose, down_pose]):
                # b x [0, 1], b x [1, 0]
                mode = m * torch.ones(B, 1)

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                num_images = input.size(0)
                # measure accuracy and record loss
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                 target.cpu().numpy())
                accAMer.update(avg_acc, cnt)

                # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                # todo 被遮挡者结果权重衰减？
                if m == 1:
                    score = meta['score'].numpy()*config.TEST.DECAY_THRE
                else:
                    score = meta['score'].numpy()

                annotation_id = meta['annotation_id'].numpy()

                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

                all_preds[global_idx:global_idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[global_idx:global_idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[global_idx:global_idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[global_idx:global_idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[global_idx:global_idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[global_idx:global_idx + num_images, 5] = score
                all_boxes[global_idx:global_idx + num_images, 6] = annotation_id
                all_boxes[global_idx:global_idx + num_images, 7] = mode.detach().cpu().numpy().reshape(-1)

                image_path.extend(meta['image'])

                global_idx += num_images

                if config.LOG:
                    msg = 'Accuracy {acc.val:.8f}'.format(acc=accAMer)
                    val_loader.set_description(msg)

                # 可视化结果
                if config.DEBUG.DEBUG is True and i % config.PRINT_FREQ == 0:
                    save_size = min(8, B)
                    meta['pred_joints_vis'] = torch.ones_like(meta['joints_vis'])

                    prefix = str(i) + ['up', 'down'][m]

                    save_debug_images(config, input[:save_size, [2,1,0], :, :], meta, target[:save_size], (pred*4)[:save_size], output[:save_size],
                                      visualize_dir, prefix)
        perf_indicator = 0.0
        if config.LOG:
            name_values, name_values_mode0, \
            name_values_mode1, name_values_mode2, \
            name_values_mode3, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path, epoch,
                filenames, imgnums
            )

            model_name = config.MODEL.NAME

            _print_name_value(name_values, 'total:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode0, 'occer:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode1, 'occed:{}'.format(model_name), log=log)

        if writer_dict and config.LOG:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_acc',
                accAMer.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    val_loader.close()
    return perf_indicator
# dcp-naive

# dcp-afi
def validate_afi(config, val_loader, val_dataset, model, output_dir, writer_dict=None,
                       epoch=-1, lambda_vals=[0, 1], log=logger, criterion=None):
    accAMer = AverageMeter()

    visualize_dir = os.path.join(output_dir, "visualize")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)

    # switch to evaluate mode
    model.eval()

    num_outputs = len(lambda_vals)*len(val_dataset)
    all_preds = np.zeros(
        (num_outputs, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_outputs, 6+1+1)) ## update to add annotation ids and mode l=0,1
    image_path = []
    filenames = []
    imgnums = []
    global_idx = 0
    with torch.no_grad():
        val_loader = tqdm(val_loader)
        for i, (input, target, target_weight, meta, target_oced, target_weight_oced, meta_oced) in enumerate(val_loader):

            B, C, H, W = input.shape
            # get output
            input = input.cuda()
            pose_dict = model(input)
            up_pose = pose_dict['up']
            down_pose = pose_dict['down']

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                pose_dict_flipped = model(input_flipped)
                up_flipped = pose_dict_flipped['up']
                down_flipped = pose_dict_flipped['down']
                # occ
                up_flipped = flip_back(up_flipped.cpu().numpy(), val_dataset.flip_pairs)
                up_flipped = torch.from_numpy(up_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    up_flipped[:, :, :, 1:] = up_flipped.clone()[:, :, :, 0:-1]
                up_pose = (up_pose + up_flipped) * 0.5

                # occee
                down_flipped = flip_back(down_flipped.cpu().numpy(), val_dataset.flip_pairs)
                down_flipped = torch.from_numpy(down_flipped.copy()).cuda()

                if config.TEST.SHIFT_HEATMAP:
                    down_flipped[:, :, :, 1:] = down_flipped.clone()[:, :, :, 0:-1]

                down_pose = (down_pose + down_flipped) * 0.5
            # --------------
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            target_oced = target_oced.cuda(non_blocking=True)
            target_weight_oced = target_weight_oced.cuda(non_blocking=True)

            loss_up_pose = criterion(up_pose, target, target_weight, target_oced)
            loss_down_pose = criterion(down_pose, target_oced, target_weight_oced, target)
            # --------------
            for m, output in enumerate({up_pose, down_pose}):
                # b x [0, 1], b x [1, 0]
                mode = m * torch.ones(B, 1)

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)


                num_images = input.size(0)
                # measure accuracy and record loss
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                 target.cpu().numpy())
                accAMer.update(avg_acc, cnt)

                # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                # todo 被遮挡者结果权重衰减？
                if m == 1:
                    score = meta['score'].numpy()*config.TEST.DECAY_THRE
                else:
                    score = meta['score'].numpy()

                annotation_id = meta['annotation_id'].numpy()

                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

                all_preds[global_idx:global_idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[global_idx:global_idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[global_idx:global_idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[global_idx:global_idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[global_idx:global_idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[global_idx:global_idx + num_images, 5] = score
                all_boxes[global_idx:global_idx + num_images, 6] = annotation_id
                all_boxes[global_idx:global_idx + num_images, 7] = mode.detach().cpu().numpy().reshape(-1)

                image_path.extend(meta['image'])

                global_idx += num_images

                if config.LOG:
                    msg = 'Accuracy {acc.val:.8f}'.format(acc=accAMer)
                    val_loader.set_description(msg)

                # 可视化结果
                if config.DEBUG.DEBUG is True and i % config.PRINT_FREQ == 0:
                    save_size = min(8, B)
                    meta['pred_joints_vis'] = torch.ones_like(meta['joints_vis'])

                    prefix = str(i) + ['up', 'down'][m]

                    save_debug_images(config, input[:save_size, [2,1,0], :, :], meta, target[:save_size], (pred*4)[:save_size], output[:save_size],
                                      visualize_dir, prefix)
        logger.info("========> interference point count: {}".format(criterion.ipc))
        perf_indicator = 0.0
        if config.LOG:
            name_values, name_values_mode0, \
            name_values_mode1, name_values_mode2, \
            name_values_mode3, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path, epoch,
                filenames, imgnums
            )

            model_name = config.MODEL.NAME

            _print_name_value(name_values, 'total:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode0, 'occer:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode1, 'occed:{}'.format(model_name), log=log)

        if writer_dict and config.LOG:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_acc',
                accAMer.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    val_loader.close()
    return perf_indicator
# dcp-afi

# --------------------------------------------------------------------------------
def validate_lambda(config, val_loader, val_dataset, model, criterion_lambda, criterion, output_dir,
             tb_log_dir, writer_dict, epoch=-1, print_prefix=''):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target_a, target_weight_a, meta_a) in enumerate(val_loader):

            import copy
            target_b = copy.deepcopy(target_a)
            target_weight_b = copy.deepcopy(target_weight_a)
            meta_b = copy.deepcopy(meta_a)

            B, C, H, W = input.shape

            for lambda_a_val in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                lambda_a = lambda_a_val*torch.ones(B, 1).cuda()
                lambda_b = 1 - lambda_a
                lambda_vec = torch.cat([lambda_a, lambda_b], dim=1)

                # compute output
                outputs = model(input, lambda_vec)
                output = outputs

                # print('lambda_a:{} out:{}, mu:{}, sigma:{}'.format(lambda_a_val, output.sum(), mu.sum(), sigma.sum()))
                
                target_a = target_a.cuda(non_blocking=True)
                target_weight_a = target_weight_a.cuda(non_blocking=True)

                target_b = target_b.cuda(non_blocking=True)
                target_weight_b = target_weight_b.cuda(non_blocking=True)

                loss_a_lambda = criterion_lambda(output, target_a, target_weight_a)
                loss_b_lambda = criterion_lambda(output, target_b, target_weight_b)

                loss = (loss_a_lambda*lambda_a.view(-1)).mean() + (loss_b_lambda*lambda_b.view(-1)).mean()

                num_images = input.size(0)
                # measure accuracy and record loss
                losses.update(loss.item(), num_images)
                _, avg_acc, cnt, pred_a = accuracy(output.cpu().numpy(),
                                                 target_a.cpu().numpy())

                _, avg_acc, cnt, pred_b = accuracy(output.cpu().numpy(),
                                                 target_b.cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if (i % config.PRINT_FREQ == 0) or (i == (len(val_loader)-1)):
                    save_size = 16
                    msg = 'Test: [{0}/{1}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Loss {loss.val:.6f} ({loss.avg:.6f})\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                          'lambda_a {lambda_val:.3f}\tout {out:.3f}'.format(
                              i, len(val_loader)-1, batch_time=batch_time,
                              loss=losses, acc=acc, lambda_val=lambda_a_val, out=output.sum()
                            )
                    logger.info(msg)

                    prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'val'), epoch, i, print_prefix)
                    suffix = 'a'
                    for count in range(min(save_size, len(lambda_a))):
                        suffix += '_[{}:{}]'.format(count, round(lambda_a[count].item(), 2))

                    save_debug_images(config, input[:save_size, [2,1,0], :, :], meta_a, target_a[:save_size], (pred_a*4)[:save_size], output[:save_size],
                                      prefix, suffix)

                    prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'val'), epoch, i, print_prefix)
                    suffix = 'b'
                    for count in range(min(save_size, len(lambda_b))):
                        suffix += '_[{}:{}]'.format(count, round(lambda_b[count].item(), 2))

                    save_debug_images(config, input[:save_size, [2,1,0], :, :], meta_b, target_b[:save_size], (pred_b*4)[:save_size], output[:save_size],
                                      prefix, suffix)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar(
            'valid_loss',
            losses.avg,
            global_steps
        )
        writer.add_scalar(
            'valid_acc',
            acc.avg,
            global_steps
        )
        writer_dict['valid_global_steps'] = global_steps + 1
        
    return 0


# markdown format output
def _print_name_value(name_value, full_arch_name, log=logger):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    log.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    log.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    log.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.4f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
def validate_lambda_oncpu(config, val_loader, val_dataset, model, criterion, output_dir,
                                 tb_log_dir, writer_dict=None, epoch=-1, print_prefix='', lambda_vals=[0, 1], log=logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # switch to evaluate mode
    model.eval()

    num_samples = len(lambda_vals)*len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6+1+1)) ## update to add annotation ids and mode l=0,1
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        val_loader = tqdm(val_loader)
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            B, C, H, W = input.shape

            for lambda_idx, lambda_val in enumerate(lambda_vals):
                # b x [0, 1], b x [1, 0]
                lambda_a = lambda_val*torch.ones(B, 1)#.cuda()
                lambda_vec = torch.cat([lambda_a, 1 - lambda_a], dim=1)

                input = input#.cuda()
                outputs = model(input, lambda_vec)
                output = outputs

                if config.TEST.FLIP_TEST:
                    input_flipped = input.flip(3)
                    outputs_flipped = model(input_flipped, lambda_vec)
                    output_flipped = outputs_flipped

                    output_flipped = flip_back(output_flipped.cpu().numpy(),
                                               val_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy())#.cuda()


                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                loss = criterion(output, target, target_weight)

                num_images = input.size(0)
                # measure accuracy and record loss
                losses.update(loss.item(), num_images)
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                 target.cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()

                c = meta['center'].numpy()
                s = meta['scale'].numpy()

                if lambda_val == 0:
                    score = meta['score'].numpy()*config.TEST.DECAY_THRE

                else:
                    score = meta['score'].numpy()

                annotation_id = meta['annotation_id'].numpy()

                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                all_boxes[idx:idx + num_images, 6] = annotation_id
                all_boxes[idx:idx + num_images, 7] = lambda_a.detach().cpu().numpy().reshape(-1)

                image_path.extend(meta['image'])

                idx += num_images

                if config.LOG:
                    msg = 'Loss {loss.val:.6f} ' \
                          'Accuracy {acc.val:.3f} ' \
                          'Lambda_a {lambda_val:.3f}'.format(
                        loss=losses, acc=acc, lambda_val=lambda_val)
                    val_loader.set_description(msg)

                # if ((i % config.PRINT_FREQ == 0) or (i == (len(val_loader)-1))) and config.LOG:
                #     save_size = min(16, B)
                # meta['pred_joints_vis'] = torch.ones_like(meta['joints_vis'])

                # prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'val'), epoch, i, print_prefix)
                # suffix = 'a'
                # for count in range(min(save_size, len(lambda_a))):
                #     suffix += '_[{}:{}]'.format(count, round(lambda_a[count].item(), 2))

                # save_debug_images(config, input[:save_size, [2,1,0], :, :], meta, target[:save_size], (pred*4)[:save_size], output[:save_size],
                #                   prefix, suffix)

        perf_indicator = 0.0
        if config.LOG:
            name_values, name_values_mode0, \
            name_values_mode1, name_values_mode2, \
            name_values_mode3, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path, epoch,
                filenames, imgnums
            )

            model_name = config.MODEL.NAME

            _print_name_value(name_values, 'total:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode0, 'occer:{}'.format(model_name), log=log)
            _print_name_value(name_values_mode1, 'occed:{}'.format(model_name), log=log)

        if writer_dict and config.LOG:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    val_loader.close()
    return perf_indicator