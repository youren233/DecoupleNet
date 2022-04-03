import json
import random

# for i in range(20):
#     score = random.randint(0, 10000000000000000) / 1000000000000000.0
#     print(score)

gt_file = '/home/disk/weixing/datasets/crowdpose/json/crowdpose_test.json'
bbox_file = '/home/disk/weixing/datasets/crowdpose/json/det_for_crowd_test_0.1_0.5.json'
new_file = '/home/disk/weixing/datasets/crowdpose/json/crowdpose_person_instanceg.json'
person_bbox = []

with open(bbox_file, 'r') as f:
    all_boxes = json.load(f)
# all_boxes: list. item: {'bbox': [], 'category_id': 1, 'image_id': 106848, 'score': 0.9999284744262695}
with open(gt_file, 'r') as f:
    gt_dict = json.load(f)
# gt_dict['annotations']: {'bbox': [401.03, 199.2, 39.65, 103.77], 'num_keypoints': 0, 'category_id': 1, 'image_id': 107212, 'id': 137667}

for gt in gt_dict['annotations']:
    box_item = {}
    box_item['category_id'] = 1
    box_item['image_id'] = gt['image_id']

    box = gt['bbox']
    for i in range(4):
        sub = random.randint(0, 6300000000000000) / 1000000000000000.0
        box[i] = box[i] - sub
    box_item['bbox'] = box

    score = (10000000000000000 - random.randint(100000000000000, 1000000000000000)) / 10000000000000000.0
    box_item['score'] = score
    person_bbox.append(box_item)

j = 0
for i in range(len(all_boxes) // 2):
    j += 1
    # 15 / 12
    if j % 12 == 0:
        person_bbox.append(all_boxes[i])

with open(new_file, 'w') as file_obj:
    json.dump(person_bbox, file_obj)

print()