import cv2
from pycocotools.coco import COCO
import numpy as np

NUM_KPTS = 14
SKELETON = [[0, 2], [1, 3], [2, 4], [3, 5], [6, 8], [8, 10], [7, 9], [9, 11], [12, 13], [0, 13], [1, 13],
            [6,13],[7, 13]]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.ndim == 2 and keypoints.shape[0] == NUM_KPTS and keypoints.shape[1] >= 2
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        # cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        # cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        if x_a > 0 and y_a > 0 and x_b > 0 and y_b > 0:
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), 255, 1)

# 从 coco读 [img: gt]
train_file = '/home/disk/weixing/datasets/crowdpose/json/crowdpose_train.json'
# val_file = '/home/disk/weixing/datasets/crowdpose/json/crowdpose_val.json'
# test_file = '/home/disk/weixing/datasets/crowdpose/json/crowdpose_test.json'

imgDir = '/home/disk/weixing/datasets/crowdpose/images/'
sktImgDir = '/home/disk/weixing/datasets/crowdpose/images_anns_skt/'

coco = COCO(train_file)
imgIds = coco.getImgIds()

imgToAnns = coco.imgToAnns

# 为每张图画骨架
for imgId in imgIds:
    anns = imgToAnns[imgId]
    oriImg = cv2.imread(imgDir + str(imgId) + '.jpg')
    for ann in anns:
        # 创建空 image
        img = np.zeros(oriImg.shape[0:2], np.uint8)

        # list len == 42
        kps = ann['keypoints']
        # reshape为 [14, 2]
        keypoints = np.zeros([14, 2])
        keypoints[:, 0] = kps[0::3]
        keypoints[:, 1] = kps[1::3]
        draw_pose(keypoints, img)
        cv2.imwrite(sktImgDir + str(imgId) + '_' + str(ann['id']) + '_skt.jpg', img)
    # cv2.imwrite(sktImgDir + str(imgId) + '.jpg', oriImg)

print('finished!')
