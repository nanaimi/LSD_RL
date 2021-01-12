import cv2
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from unrealcv import client
from unrealcv.util import read_npy, read_png

from io import BytesIO
import PIL.Image

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torchsummary as summary

model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
model.eval()

feature_extractor = nn.Sequential(*(list(model.children())[0]))

mobilenet = models.mobilenet_v2(pretrained=True)

features = nn.Sequential(*(list(mobilenet.children())[0]))

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Fix the parameters of the feature extractor:
for param in features.parameters():
    param.requires_grad = False

client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game from http://unrealcv.github.io first.')
else:
    print('UnrealCV server is running.')

time.sleep(5)
res = client.request('vget /camera/0/lit png')

print("####################### reading with read_png #######################")
img_read = read_png(res)
print(type(img_read))
print(img_read.shape)
print("###################### reading with decode_png ######################")
img_decode =  PIL.Image.open(BytesIO(res))
print(type(img_decode))



img_tensor = preprocess(img_decode)
print(img_tensor.shape)
img_tensor.unsqueeze(0)
print(img_tensor.shape)

img_variable = Variable(img_tensor)
fc_out = feature_extractor(img_variable)
print(fc_out)
# Create figure and axes
# fig,ax = plt.subplots(1,4)


print("######################### Shape of image #########################")
# print(img.shape)
# empty = np.zeros((480, 640, 4))
# Display the image
# ax[0].imshow(empty)
# ax[1].imshow(img_read)
# resized_image = img[...,:3]
# ax[2].imshow(img_decode)
# swapped = np.moveaxis(resized_image, 2, 0)
# ax[3].imshow(preprocess(img_read))
# print(type(img))
# plt.show()
# swapped = np.moveaxis(resized_image, 2, 0)
# print(resized_image.shape)
# print(swapped.shape)
# tensor = torch.from_numpy(swapped)
# print(tensor.shape)
# print("####################### Predicted features #######################")
# print(mobilenet(tensor))
