import json


js_file = '/home/disk/weixing/datasets/crowdpose/json/crowdpose_train.json'
js_test = json.load(open(js_file, 'r'))

from demo_gcn import draw_pose

img_path = '/home/disk/weixing/datasets/crowdpose/images/100486.jpg'
import cv2
import numpy as np

img = cv2.imread(img_path)

for ann_dic in js_test['annotations']:
    if ann_dic['image_id'] == 100486:
        kpts = ann_dic['keypoints']
        kpts = np.resize(kpts, [14, 3])

        draw_pose(kpts, img)

cv2.imwrite('sdfasdfas.jpg', img)
print('over')
