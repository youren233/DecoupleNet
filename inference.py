from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

import _init_paths
import lib.models
from lib.config import cfg
from lib.config import update_config
from lib.core.function import get_final_preds
from lib.utils.transforms import get_affine_transform

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# SKELETON = [
#     [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
# ]

SKELETON = [[0, 2], [1, 3], [2, 4], [3, 5], [6, 8], [8, 10], [7, 9], [9, 11], [12, 13], [0, 13], [1, 13],
             [6,13],[7, 13]]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 14

CTX = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='experiments/crowdpose/hrnet/test.yaml')
    parser.add_argument('--video', type=str)
    parser.add_argument('--webcam',action='store_true')
    parser.add_argument('--image', type=str, default="input")
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--showFps',action='store_true')
    parser.add_argument('--local_rank',action='store_true')
    parser.add_argument('--exp_id',action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase  
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    # pose_model = eval('lib.models.'+cfg.MODEL.NAME+'.get_pose_net')(
    #     cfg, is_train=False
    # )
    pose_model = lib.models.pose_hrnet_decouple_stupid.get_pose_net(cfg, is_train=False)

    # 模型加载。模型路径手动写
    # model_file = 'output/Two_8_cnn_arm_mse_triplet_w32_256x192-two-arm_test/checkpoint_208.pth'
    model_file = 'output/train_two_2_64/checkpoint_207.pth'
    # model_file = 'output/train-two_2blocks_64channels_att_test/checkpoint_209.pth'
    # model_file = os.path.join(ckpt_dir, cfg.TEST.MODEL_FILE)
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(model_file))
        model_object = torch.load(model_file, map_location='cpu')
        if 'latest_state_dict' in model_object.keys():
            print('=> loading from latest_state_dict at {}'.format(model_file))
            pose_model.load_state_dict(model_object['latest_state_dict'])
        elif 'state_dict' in model_object.keys():
            print('=> loading from latest_state_dict at {}'.format(model_file))
            pose_model.load_state_dict(model_object['state_dict'])
        else:
            print('=> no latest_state_dict found')
            pose_model.load_state_dict(model_object)
    else:
        model_state_file = os.path.join(
            ckpt_dir, 'final_state.pth'
        )
        print('=> loading model from {}'.format(model_state_file))
        pose_model.load_state_dict(torch.load(model_state_file, map_location='cpu'))

    pose_model.to(CTX)
    pose_model.eval()

    # 视频、webcam
    image_bgrs = []
    if args.webcam:
        vidcap = cv2.VideoCapture(0)
    elif args.video:
        vidcap = cv2.VideoCapture(args.video)
    elif args.image:
        print('=> images')
    else:
        print('please use --video or --webcam or --image to define the input.')
        return

    # 视频、webcam
    if args.webcam or args.video:
        if args.write:
            save_path = 'output.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_path,fourcc, 24.0, (int(vidcap.get(3)),int(vidcap.get(4))))
        while True:
            ret, image_bgr = vidcap.read()
            if ret:
                last_time = time.time()
                image = image_bgr[:, :, [2, 1, 0]]

                input = []
                img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
                input.append(img_tensor)

                # object detection box
                pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

                # pose estimation
                if len(pred_boxes) >= 1:
                    for box in pred_boxes:
                        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                        image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                        if len(pose_preds)>=1:
                            for kpt in pose_preds:
                                draw_pose(kpt,image_bgr) # draw the poses

                if args.showFps:
                    fps = 1/(time.time()-last_time)
                    img = cv2.putText(image_bgr, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                if args.write:
                    out.write(image_bgr)

                cv2.imshow('demo',image_bgr)
                if cv2.waitKey(1) & 0XFF==ord('q'):
                    break
            else:
                print('cannot load the video.')
                break

        cv2.destroyAllWindows()
        vidcap.release()
        if args.write:
            print('video has been saved as {}'.format(save_path))
            out.release()

    else:
        files = os.listdir(args.image)
        for file_name in files:
            file_path = os.path.join(args.image, file_name)
            image_bgr = cv2.imread(file_path)
            # estimate on the image
            last_time = time.time()
            image = image_bgr[:, :, [2, 1, 0]]

            input = []
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
            input.append(img_tensor)

            # object detection box
            pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

            # pose estimation
            no_box = False
            if len(pred_boxes) >= 1:
                for i, box in enumerate(pred_boxes):
                    if no_box and i >= 1:
                        break
                    tmp_img = image_bgr.copy()
                    # draw_bbox(box, image_bgr)
                    center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                    image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                    pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale, no_box=no_box)
                    if len(pose_preds)>=1:
                        for kpt in pose_preds:
                            draw_pose(kpt,image_bgr) # draw the poses
                    # file_path = os.path.join('output', 'infer', str(i) + "_" + file_name)
                    # cv2.imwrite(file_path, tmp_img)

            if args.showFps:
                fps = 1/(time.time()-last_time)
                img = cv2.putText(image_bgr, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # cv2.imshow('demo',image_bgr)
            file_path = os.path.join('output', 'infer', file_name)
            cv2.imwrite(file_path, image_bgr)
            # if cv2.waitKey(0) & 0XFF==ord('q'):
            #     cv2.destroyAllWindows()


def draw_pose(keypoints,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    # color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(SKELETON) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    assert keypoints.shape[0] == NUM_KPTS
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]

        sktColor = (0, 0, 255)#colors[i]

        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1]
        cv2.circle(img, (int(x_a), int(y_a)), 5, sktColor, -1)
        cv2.circle(img, (int(x_b), int(y_b)), 5, sktColor, -1)

        # cv2.putText(img, str(kpt_a), (int(x_a), int(y_a)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        # cv2.putText(img, str(kpt_b), (int(x_b), int(y_b)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

        # if x_a > 0 and y_a > 0 and x_b > 0 and y_b > 0:
        #     cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), sktColor, 4, lineType=cv2.LINE_AA)

def draw_bbox(box,img):
    """draw the detected bounding box on the image.
    :param img:
    """
    x1, y1 = int(box[0][0]), int(box[0][1])
    x2, y2 = int(box[1][0]), int(box[1][1])
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0),thickness=2)


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score)<threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale, no_box=False):
    rotation = 0

    if no_box:
        model_input = cv2.resize(image, (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])))
    else:
        # pose estimation transformation
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    model_input = model_input.to(CTX)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        if isinstance(output, dict):
            preds1, _ = get_final_preds(
                cfg,
                output['up'].clone().cpu().numpy(),
                np.asarray([center]),
                np.asarray([scale]))
            preds2, _ = get_final_preds(
                cfg,
                output['down'].clone().cpu().numpy(),
                np.asarray([center]),
                np.asarray([scale]))
            # preds1[0, 10, 0] = 12.0041275
            # preds1[0, 10, 1] = 479.90295
            preds = np.concatenate((preds1, preds2))
        else:
            preds, _ = get_final_preds(
                cfg,
                output.clone().cpu().numpy(),
                np.asarray([center]),
                np.asarray([scale]))
        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


if __name__ == '__main__':
    main()
