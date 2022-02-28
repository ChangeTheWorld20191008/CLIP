import clip
import os
import torch
import numpy as np
from PIL import Image


class Accuracy():
    def __init__(self, TP=0, FP=0, FN=0) -> None:
        self.TP = TP
        self.FP = FP
        self.FN = FN

    def __getitem__(self, key):
        # We use the getattr() function to get the attribute of an object.
        # It will first try __getattribute__() function for the directly
        # accessible attributes. If an AttributeError is raised, it will
        # then try __getattr__() function, where we can customize the
        # attribute looking up procedure.
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)


coco_id_mapping = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'fire', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
    47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush',
}
# prob_thresh = 0.2

# Loading groundtruth
gt_file = '/home/zhuhao/dataset/tmp/image_label'
img_cls_map = dict()
for line in open(gt_file):
    name_and_cls = line.split(":")
    img_name = name_and_cls[0]
    cls_list = name_and_cls[-1].split("\n")[0].split(",")
    img_cls_map[img_name] = cls_list
# print(f"[TMP]: img cls info is {img_cls_map}")

# Loading the model
clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()

text_descriptions = [value for value in coco_id_mapping.values()]
# print(f"[TMP]: text_descriptions are {text_descriptions}")
text_tokens = clip.tokenize(text_descriptions).cuda()

with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

image_dir = '/home/zhuhao/dataset/public/coco/val2017'
# image_dir = '/home/zhuhao/dataset/tmp/images'
image_list = []
label_accuray_map = dict()

for image_name in os.listdir(image_dir):
    image = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
    images = [preprocess(image)]

    image_input = torch.tensor(np.stack(images)).cuda()

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

    top_prob = top_probs[0]
    top_label = top_labels[0]

    print(f"[TMP]: ====={image_name}=====")
    labels = [text_descriptions[int(index)] for index in top_label.numpy()]
    print(
        f"[TMP]: top prob are {top_prob}, top_label are {labels}")

    # dt_label_list = []
    # for i in range(len(top_prob)):
    #     if top_prob[i] >= prob_thresh:
    #         label = text_descriptions[int(top_label[i])]
    #         dt_label_list.append(label)
    dt_label_list = labels

    img_gt = img_cls_map[image_name]
    for label in dt_label_list:
        if label in label_accuray_map:
            accuray = label_accuray_map[label]
        else:
            accuray = Accuracy()

        if label in img_gt:
            accuray.TP += 1
        else:
            accuray.FP += 1

        label_accuray_map[label] = accuray

    for label in img_gt:
        if label in label_accuray_map:
            accuray = label_accuray_map[label]
        else:
            accuray = Accuracy()

        if label not in dt_label_list:
            accuray.FN += 1

        label_accuray_map[label] = accuray

# Calculate precision and recall
all_TP = 0
all_FP = 0
all_FN = 0
for label, accuray in label_accuray_map.items():
    TP = accuray.TP
    FP = accuray.FP
    FN = accuray.FN

    all_TP += TP
    all_FP += FP
    all_FN += FN

    dt_value = TP+FP
    if dt_value == 0:
        precision = 1
    else:
        precision = TP/dt_value

    gt_value = TP+FN
    if gt_value == 0:
        recall = 1
    else:
        recall = TP/gt_value

    print(
        f"[INFO]: {label}: TP: {TP}, FP: {FP}, FN: {FN}, precision: {precision}, recall: {recall}")

precision = all_TP/(all_TP+all_FP)
recall = all_TP/(all_TP+all_FN)
print(
    f"[INFO]: total: TP: {all_TP}, FP: {all_FP}, FN: {all_FN}, precision: {precision}, recall: {recall}")
