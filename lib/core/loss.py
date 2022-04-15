# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.utils.gaussian import GaussianSmoothing

from lib.core.inference import get_max_preds


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze() ## size=(B, 64 x 48)
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class JointsLambdaMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsLambdaMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze() ## size=(B, 64 x 48)
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        loss = loss.mean(dim=1) / num_joints
        return loss


class JointsExpectationLoss(nn.Module):
    def __init__(self):
        super(JointsExpectationLoss, self).__init__()
        self.criterion = nn.L1Loss(reduction='mean')
        self.gaussian_smoothing = GaussianSmoothing(channels=17, kernel_size=11, sigma=6) ## 11 copied from dark
        # output = self.gaussian_smoothing(F.pad(output, (5, 5, 5, 5), mode='reflect'))
        # self.cutoff_threshold = 0.0001234098 ##this is the min value in gt heatmaps

        return

    def forward(self, output, target_joint, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        width = output.size(3)

        device = output.get_device()

        # --------------------------------------------
        heatmaps_pred = output.split(1, 1) ## split B x 17 x 64 x 48 tensor into 17 single chunks along dim 1
        gt_joints = target_joint.split(1, 1) ## split B x 17 x 64 x 48 tensor into 17 single chunks along dim 1
        locs = torch.arange(output.size(2)*output.size(3)).to(device)

        loss = 0
        # --------------------------------------------
        for idx in range(num_joints):
            original_heatmap_pred = heatmaps_pred[idx].squeeze() ## size=(B, 64 , 48)
            gt_joint = gt_joints[idx].squeeze() ## size=(B, 2)

            heatmap_pred = original_heatmap_pred.view(batch_size, -1)

            # #---------------------------------
            # heatmap_pred = F.softmax(heatmap_pred, dim=1)
            # heatmap_pred = heatmap_pred.clamp(min=1e-10)
            # expected_pred = (heatmap_pred*locs).sum(dim=1)/heatmap_pred.sum(dim=1)

            #---------------------------------
            # heatmap_pred = F.relu(heatmap_pred)
            # heatmap_pred = heatmap_pred.clamp(min=1e-10)
            # expected_pred = (heatmap_pred*locs).sum(dim=1)/heatmap_pred.sum(dim=1)

            #---------------------------------
            heatmap_pred = heatmap_pred.clamp(min=1e-10)
            expected_pred = (heatmap_pred*locs).sum(dim=1)/25.0813 ## B
            
            #---------------------------------
            # expected_pred = expected_pred.view(batch_size, 1) ## B x 1

            # expected_pred = expected_pred.repeat(1, 2)
            # expected_pred[:, 0] = expected_pred[:, 0] % width
            # expected_pred[:, 1] = expected_pred[:, 1] / width

            # loss += self.criterion(
            #     expected_pred.mul(target_weight[:, idx]),
            #     gt_joint.mul(target_weight[:, idx])
            # )

            #---------------------------------
            expected_pred = expected_pred.view(-1, 1)
            linear_gt_joint = width*gt_joint[:, 1] + gt_joint[:, 0]
            linear_gt_joint = linear_gt_joint.view(-1, 1)

            loss += self.criterion(
                expected_pred.mul(target_weight[:, idx]),
                linear_gt_joint.mul(target_weight[:, idx])
            )

        loss = loss / num_joints
        return loss

class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]  # shape(14, )
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        # shape: (B, joints num) (32, 14)
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)

# ----------------------------------------------------------------------
class JointsOCKSMSELoss(nn.Module):
    # Online Confused Keypoint Suppression
    # skelenton = [[0, 2], [1, 3], [2, 4], [3, 5], [6, 8], [8, 10], [7, 9], [9, 11], [12, 13], [0, 13], [1, 13],
    #             [6,13],[7, 13]]

    def __init__(self, config):
        super(JointsOCKSMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = config.LOSS.USE_TARGET_WEIGHT
        self.ohkm_criterion = JointsOHKMMSELoss(self.use_target_weight)
        self.thres = config.LOSS.OKS_THRES

        self.confusedCount = 0
        self.confusedMask = []

    def computeOks(self, dtCoord, gtCoord, area, vg):
        '''
        compute oks between dtCoord and gtCoord

        dtCoord: [B, 14, 2]
        gtCoord: [B, 14, 2]
        '''
        # 边界控制
        pass

        sigmas = np.array([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79])/10.0
        vars = (sigmas * 2)**2
        k = len(sigmas)

        # create bounds for ignore regions(double the gt bbox)
        xg = gtCoord[:, 0]
        yg = gtCoord[:, 1]
        # vg = g[2::3]
        # k1 = np.count_nonzero(vg > 0)
        # if k1 == 0: # impossible
        #     return iou

        xd = dtCoord[:, 0]
        yd = dtCoord[:, 1]
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg

        tmparea = area * 0.53
        e = (dx**2 + dy**2) / vars / (tmparea+np.spacing(1)) / 2

        e = e.numpy()
        ious = np.exp(-e)
        ious[vg == 0] = 0
        return ious
    
    def getConfusedMask(self, pred, target_coord, another_coord, scale, joints_vis):
        area = scale[0] * 160 * scale[1] * 160
        # 找到预测错误的点
        iou = self.computeOks(pred, target_coord, area, joints_vis[:, 0])
        isFalse = iou < self.thres

        # 计算是否与 another_coord接近
        iou = self.computeOks(pred, another_coord, area, joints_vis[:, 0])
        isConfused = iou > self.thres
        confusedMask = isFalse & isConfused

        return np.count_nonzero(confusedMask), confusedMask

    def ocks(self, loss, output, target, another_target, meta):
        # heatmap to point
        pred, _ = get_max_preds(output.detach().cpu().numpy())    # [batch, 14, 2]
        target_coord, _ = get_max_preds(target.detach().cpu().numpy())
        another_coord, _ = get_max_preds(another_target.detach().cpu().numpy())

        ocks_loss = 0.

        # 遍历 batch
        for i in range(loss.size()[0]):
            sub_loss = loss[i]  # shape(14, )
            ocks_loss += torch.sum(sub_loss)
            # get the false confused point idx
            num, confusedMask = self.getConfusedMask(pred[i], target_coord[i], another_coord[i],
                                                     meta['scale'][i], meta['joints_vis'][i])
            # debug
            self.confusedMask = confusedMask
            self.confusedCount += num

            tmp_loss = sub_loss[confusedMask]
            if num > 0:
                ocks_loss += torch.sum(tmp_loss) / num
        ocks_loss /= loss.size()[0]
        return ocks_loss

    def forward(self, output, target, another_target, target_weight, meta):
        ohkm_loss = self.ohkm_criterion(output, target, target_weight)
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)   # shape: B, joints num. (32, 14)

        return ohkm_loss + self.ocks(loss, output, target, another_target, meta)

# ----------------------------------------------------------------------
class JointsTripletMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsTripletMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight

        # interference point count
        self.ipc = 0

    def tripletLoss(self, loss, another_loss):
        triple_loss = loss - another_loss
        triple_loss[triple_loss < 0] = 0

        # debug
        tmp = triple_loss.detach().cpu().numpy()
        self.ipc += np.count_nonzero(tmp)

        return torch.mean(triple_loss, dim=1)

    def forward(self, output, target, target_weight, another_target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        another_heatmaps_gt = another_target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        another_loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            another_heatmap_gt = another_heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
                another_loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    another_heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(0.5 * self.criterion(heatmap_pred, heatmap_gt))
                another_loss.append(0.5 * self.criterion(heatmap_pred, another_heatmap_gt))

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        # shape: (B, joints num) (32, 14)
        loss = torch.cat(loss, dim=1)

        another_loss = [l.mean(dim=1).unsqueeze(dim=1) for l in another_loss]
        # shape: (B, joints num) (32, 14)
        another_loss = torch.cat(another_loss, dim=1)
        triplet_loss = torch.mean(loss, dim=1) + self.tripletLoss(loss, another_loss)
        return torch.mean(triplet_loss)

class ProMaskLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, output, pro_mask):
        # L1 and L2 distance
        distL1 = pro_mask - output
        distL2 = distL1 ** 2

        regionPos = (pro_mask > 0).float()
        regionNeg = (pro_mask == 0).float()
        sumPos = torch.sum(regionPos)
        sumNeg = torch.sum(regionNeg)

        weightPos = sumNeg / (sumPos + sumNeg).float() * regionPos
        weightNeg = sumPos / (sumPos + sumNeg).float() * regionNeg
        ctx.save_for_backward(distL1, weightPos, weightNeg)
        return torch.sum(distL2 * (weightPos + weightNeg)) / (pro_mask.shape[0]) / 2 / torch.sum(
            weightNeg + weightPos)

    @staticmethod
    def backward(ctx, loss_output):
        distL1, weightPos, weightNeg = ctx.saved_tensors

        return -distL1 * (weightPos + weightNeg) / 1, None

if __name__ == "__main__":
    # unit test
    import torch

    output = torch.full([16, 14, 64, 48], 0.1) # B, joints num, 64, 48

    target = torch.rand([16, 14, 64, 48]) # max: 0.2191; min: -0.1087
    another_target = torch.rand([16, 14, 64, 48]) # B, joints num, 64, 48
    target_weight = torch.full([16, 14, 1], 10)

    # criterion = JointsOCKSMSELoss(False)
    # criterion = JointsOHKMMSELoss(False)
    criterion = JointsTripletMSELoss(True)

    loss = criterion(output, target, target_weight, another_target) #, another_pose
    print("loss: {}\nshape:{}".format(loss, loss.shape))
