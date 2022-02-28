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
    'person': 'person',
    'transportation': 'bicycle,car,motorcycle,airplane,bus,train,truck,boat',
    'traffic sign': 'traffic light,stop sign',
    'fire hydrant': 'fire hydrant',
    'parking meter': 'parking meter',
    'furniture': 'bench,chair,couch,bed,dining table',
    'animal': 'bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe',
    'carry-on items': 'backpack,umbrella,handbag,tie,suitcase',
    'sport equipment': 'frisbee,skis,snowboard,sports ball,kite,baseball bat,\
        baseball glove,skateboard,surfboard,tennis racket',
    'tableware': 'bottle,wine glass,cup,fork,knife,spoon,bowl',
    'fruit': 'banana,apple,orange',
    'food': 'sandwich,hot dog,pizza,donut,cake',
    'plant': 'broccoli,carrot,potted plant',
    'room': 'toilet,sink',
    'electronic product': 'tv,laptop,mouse,remote,keyboard,cell phone,\
        microwave,oven,toaster,refrigerator,hair drier',
    'item': 'book,clock,vase,scissors,teddy bear,toothbrush'
}
label_mapping = dict()
for b_cat, s_cat in coco_id_mapping.items():
    s_cat_list = s_cat.split(',')
    label_mapping[b_cat] = s_cat_list

prob_thresh = 0.2

# Loading groundtruth
gt_file = '/home/zhuhao/dataset/tmp/image_label'
img_cls_map = dict()
for line in open(gt_file):
    name_and_cls = line.split(":")
    img_name = name_and_cls[0]
    cls_list = name_and_cls[-1].split("\n")[0].split(",")

    cls_set = set()
    for category in cls_list:
        for b_cat, s_cat_list in label_mapping.items():
            if category in s_cat_list:
                cls_set.add(b_cat)
    img_cls_map[img_name] = cls_set
# print(f"[TMP]: img cls info is {img_cls_map}")

# Loading the model
clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()

text_descriptions = [value for value in label_mapping.keys()]
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

    dt_label_list = []
    for i in range(len(top_prob)):
        if top_prob[i] >= prob_thresh:
            label = text_descriptions[int(top_label[i])]
            dt_label_list.append(label)
        else:
            break

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
