import json
import os
import cv2
import numpy as np
from inference import draw_pose

train_json = '/home/disk/weixing/datasets/crowdpose/json/crowdpose_train.json'
val_json = '/home/disk/weixing/datasets/crowdpose/json/crowdpose_val.json'
test_json = '/home/disk/weixing/datasets/crowdpose/json/crowdpose_test.json'
img_path = '/home/disk/weixing/pycharm/DecoupleNet/input'

# 解析所有 json文件
jsList = []
for js in [train_json, val_json, test_json]:
    jsList.append(json.load(open(js, 'r')))
# 拿到所有待画pose 的图像
files = os.listdir(img_path)
imgIds = [file_name[:-4] for file_name in files]
# 结果字典
img_dict = {}

for js in jsList:
    annos = js['annotations']
    for ann_dic in annos:
        imgId = str(ann_dic['image_id'])
        if imgId in imgIds:

            img_name = imgId + ".jpg"
            if img_name not in img_dict.keys():
                img_file = os.path.join(img_path, img_name)
                img = cv2.imread(img_file)
                img_dict[img_name] = img
            else:
                img = img_dict[img_name]
            kpts = ann_dic['keypoints']
            kpts = np.resize(kpts, [14, 3])

            draw_pose(kpts, img)

for img_name, imgNp in img_dict.items():
    file_path = os.path.join("output/infer", img_name)
    cv2.imwrite(file_path, imgNp)
print('over')
