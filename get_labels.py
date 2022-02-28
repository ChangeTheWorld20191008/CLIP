import os
import clip
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
# from torchvision.datasets import CIFAR100
from PIL import Image


# Loading the model
clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()

# Setting up input images
original_images = []
images = []

image_dir = '/home/zhuhao/dataset/public/coco/val2017'
image_list = []
for image_name in os.listdir(image_dir):
    if image_name.endswith(".png") or image_name.endswith(".jpg"):
        image_list.append(image_name)

sub_img_list = random.sample(image_list, 8)

# plt.figure(figsize=(16, 5))
for filename in sub_img_list:
    name = os.path.splitext(filename)[0]
    print(f"[INFO]: file name is {name}")

    image = Image.open(os.path.join(image_dir, filename)).convert("RGB")

    # plt.subplot(2, 4, len(images) + 1)
    # plt.imshow(image)
    # plt.xticks([])
    # plt.yticks([])

    original_images.append(image)
    images.append(preprocess(image))

# plt.tight_layout()

# Building features
image_input = torch.tensor(np.stack(images)).cuda()

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)

# Zero-Shot Image Classification
# cifar100 = CIFAR100(
#     os.path.expanduser("~/.cache"), transform=preprocess, download=True)
# text_descriptions = cifar100.classes

text_descriptions = [
    'person', 'vehicle', 'animal', 'fruit', 'airplane', 'furniture',
    'traffic sign', 'food']
text_tokens = clip.tokenize(text_descriptions).cuda()

with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

plt.figure(figsize=(16, 16))

for i, image in enumerate(original_images):
    plt.subplot(4, 4, 2 * i + 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(4, 4, 2 * i + 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    plt.barh(y, top_probs[i])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(
        y, [text_descriptions[index] for index in top_labels[i].numpy()])
    plt.xlabel("probability")

plt.subplots_adjust(wspace=0.5)
plt.show()
