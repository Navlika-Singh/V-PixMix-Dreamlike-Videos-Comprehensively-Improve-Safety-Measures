import argparse
import os
import shutil
import time

import pixmix_utils as utils
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import torchvision.transforms.functional as TF

from PIL import Image
import cv2

k = 4
beta = 4
all_ops = True
aug_severity = 1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
standard_preprocess = {'tensorize': to_tensor, 'normalize': normalize}


def pixmix(orig, mixing_pic, preprocess):
    
    mixings = utils.mixings
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig))
    else:
        mixed = tensorize(orig)
    
    for _ in range(np.random.randint(k + 1)):
        
        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(orig))
        else:
            aug_image_copy = tensorize(mixing_pic)

        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, beta)
        mixed = torch.clip(mixed, 0, 1)

    # return normalize(mixed)
    return mixed

def augment_input(image):
    aug_list = utils.augmentations_all if all_ops else utils.augmentations
    op = np.random.choice(aug_list)
    return op(image.copy(), aug_severity)


# Load the image
# sample_image = Image.open('panda.jpg')
mixing_image = Image.open("fractal.jpg")

# sample_image = sample_image.resize((256,256))
mixing_image = mixing_image.resize((256,256))

# # Apply the pixmix
# pixed = pixmix(sample_image, mixing_image, standard_preprocess)

# image = TF.to_pil_image(pixed)
# image.save('pixed_panda.jpg')

# a = to_tensor(sample_image)
# b = to_tensor(mixing_image)
# res = utils.multiply(a,b, beta)
# res = torch.clip(res, 0, 1)
# res = TF.to_pil_image(res)
# res.save('pixed_panda.jpg')

# apply pixmix to a video
video_path = "v_Archery_g01_c04.avi"
output_path = "output.avi"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

for i in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize((256,256))
        # pixed = pixmix(frame, mixing_image, standard_preprocess)

        a = to_tensor(frame)
        b = to_tensor(mixing_image)
        if np.random.random() < 0.5:
            pixed = utils.add(a,b, beta)
        else:
            pixed = utils.multiply(a,b, beta)
        pixed = torch.clip(pixed, 0, 1)

        pixed = TF.to_pil_image(pixed)
        pixed = cv2.cvtColor(np.array(pixed), cv2.COLOR_RGB2BGR)
        pixed = cv2.resize(pixed, (frame_width, frame_height))
        out.write(pixed)
    else:
        break

cap.release()
out.release()
