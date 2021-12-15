# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torchvision
import torch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2

from lib.core.inference import get_max_preds
from lib.utils.utils import batch_unnormalize_image

import numpy as np
from matplotlib import pyplot as plt
def vis_tensor_img(tensor):
    img = np.transpose(tensor.cpu().numpy(), (1, 2, 0))
    plt.imshow(img)
    plt.show()

def vis_keypoint(kpts, img_name, color=(255,128,128), thr=0.5):
    img = cv2.imread(img_name)
    # kpts = np.array(kpts).reshape(-1,3)
    skelenton = [[0, 2], [1, 3], [2, 4], [3, 5], [6, 8], [8, 10], [7, 9], [9, 11], [12, 13], [0, 13], [1, 13],
                 [6,13],[7, 13]]
    points_num = [num for num in range(14)]
    for sk in skelenton:
        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1] , 1]))
        if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0:
            cv2.line(img, pos1, pos2, color, 2, 8)
    for points in points_num:
        pos = (int(kpts[points,0]),int(kpts[points,1]))
        if pos[0] > 0 and pos[1] > 0:
            cv2.circle(img, pos,10,(0,255,255),-1) #为肢体点画红色实心圆 return img

    plt.imshow(img)
    plt.show()

def save_batch_image(batch_image, file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()
    cv2.imwrite(file_name, ndarr)
    return


def save_pretty_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''

    B, C, H, W = batch_image.size()

    if isinstance(batch_joints, torch.Tensor):
        batch_joints = batch_joints.cpu().numpy()

    batch_image = batch_unnormalize_image(batch_image)

    grid = []

    for i in range(B):
        image = batch_image[i].permute(1, 2, 0).cpu().numpy() #image_size x image_size x RGB
        image = image.copy()
        kps = batch_joints[i]
        kps = np.concatenate((kps, np.zeros((17, 1))), axis=1)
        kp_vis_image = coco_vis_keypoints(image, kps, alpha=0.7) ## H, W, C
        kp_vis_image = kp_vis_image.transpose((2, 0, 1)).astype(np.float32)
        kp_vis_image = torch.from_numpy(kp_vis_image.copy())
        grid.append(kp_vis_image)

    grid = torchvision.utils.make_grid(grid, nrow, padding)
    ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(file_name, ndarr)
    return

def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0

    if isinstance(batch_joints, torch.Tensor):
        batch_joints = batch_joints.cpu().numpy()
        batch_joints_vis = batch_joints_vis.cpu().numpy()

    batch_joints = batch_joints.copy() 

    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]            

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]

                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)



def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    ## normalize image
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)

def save_batch_heatmaps_one(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    ## normalize image
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)

    heatmap_height = batch_heatmaps.size(2)*4*2
    heatmap_width = batch_heatmaps.size(3)*4*2


    grid_image = np.zeros((batch_size*heatmap_height,
                           heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)

        total_heatmap = np.zeros((heatmap_height, heatmap_width, 3), dtype=np.uint8)

        # remove_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        # remove_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,]
        ALL_KP_ORDER = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
            'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
        # remove_joints = ['nose', 'left_eye','right_eye','left_ear','right_ear','left_shoulder', 'left_hip','right_hip']
        # remove_joints = ['nose', 'left_eye','right_eye','left_ear','right_ear','left_hip','right_hip']
        remove_joints = ['nose', 'left_eye','right_eye','left_ear','right_ear', 'left_elbow', 'right_elbow', 'left_wrist','right_wrist',]

        remove_joints_idx = [ALL_KP_ORDER.index(name) for name in remove_joints]
        heatmaps[remove_joints_idx, :, :] = 0

        raw_total_heatmap = heatmaps.sum(axis=0).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(raw_total_heatmap, cv2.COLORMAP_JET)
        colored_heatmap = cv2.resize(colored_heatmap, (int(heatmap_width), int(heatmap_height)))

        # masked_image = colored_heatmap*0.7 + resized_image*0.3
        masked_image = colored_heatmap*0.6 + resized_image*0.4

        grid_image[height_begin:height_end, 0:heatmap_width, :] = masked_image
            
    cv2.imwrite(file_name, grid_image)

def save_image(batch_image, file_name, nrow=8, padding=2):
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    image_width = 384
    image_height = 512

    ndarr = cv2.resize(ndarr, (image_width, image_height))

    cv2.imwrite(file_name, ndarr)
    return

def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix, suffix=''):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt_{}.jpg'.format(prefix, suffix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        if 'pred_joints_vis' in meta.keys():
            save_batch_image_with_joints(
                input, joints_pred, meta['pred_joints_vis'],
                '{}_pred_{}.jpg'.format(prefix, suffix)
            )
        else:            
            save_batch_image_with_joints(
                input, joints_pred, meta['joints_vis'],
                '{}_pred_{}.jpg'.format(prefix, suffix)
            )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt_{}.jpg'.format(prefix, suffix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred_{}.jpg'.format(prefix, suffix)
        )

    # # # ----------------------------------------------------
    # # ## for paper only
    # ## all heatmaps in one image
    # save_batch_heatmaps_one(input, output, '{}_hm_in_one_pred_{}.jpg'.format(prefix, suffix))
    # save_batch_heatmaps_one(input, target, '{}_hm_in_one_gt_{}.jpg'.format(prefix, suffix))
    # save_image(input, '{}_rgb_{}.jpg'.format(prefix, suffix))


    # ----------------------------------------------------

    return

def save_pretty_debug_images(config, input, meta, target, joints_pred, output,
                      prefix, suffix=''):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_pretty_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt_{}.jpg'.format(prefix, suffix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_pretty_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred_{}.jpg'.format(prefix, suffix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt_{}.jpg'.format(prefix, suffix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred_{}.jpg'.format(prefix, suffix)
        )


# ------------------------------------------------------------------------------------
# standard COCO format, 17 joints
COCO_KP_ORDER = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


COCO_KP_CONNECTIONS = kp_connections(COCO_KP_ORDER)

# ------------------------------------------------------------------------------------
def coco_vis_keypoints(image, kps, alpha=0.7):
    # image is [image_size, image_size, RGB] #numpy array
    # kps is [17, 3] #numpy array
    kps = kps.astype(np.int16)
    bgr_image = image[:, :, ::-1] ##if this is directly in function call, this produces weird opecv cv2 Umat errors
    kp_image = vis_keypoints(bgr_image, kps.T) #convert to bgr
    kp_image = kp_image[:, :, ::-1] #bgr to rgb

    return kp_image

# ------------------------------------------------------------------------------------
def vis_keypoints(img, kps, kp_thresh=-1, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (3, #keypoints) where 3 rows are (x, y, depth z).
    needs a BGR image as it only uses opencv functions, returns a bgr image
    """
    dataset_keypoints = COCO_KP_ORDER
    kp_lines = COCO_KP_CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) // 2
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) // 2
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')

    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        kp_mask = cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        kp_mask = cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            kp_mask = cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            kp_mask = cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            kp_mask = cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    ## weird opencv bug on cv2UMat vs numpy
    if type(kp_mask) != type(img):
        kp_mask = kp_mask.get()

    # Blend the keypoints.
    result = cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
    return result

# -----------------------------------------------------------------------


