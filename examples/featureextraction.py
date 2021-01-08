import cv2
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from unrealcv import client
from unrealcv.util import read_npy, read_png

import torch
import torch.nn as nn
import torchvision.models as models
import torchsummary as summary


mobilenet = models.mobilenet_v2(pretrained=True)

features = nn.Sequential(*(list(mobilenet.children())[0]))
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
img = read_png(res)

# Create figure and axes
fig,ax = plt.subplots(1,3)


print("######################### Shape of image #########################")
print(img.shape)
empty = np.zeros((480, 640, 4))
# Display the image
ax[0].imshow(empty)
ax[1].imshow(img)
resized_image = img[...,:3]
ax[2].imshow(resized_image)
plt.show()
swapped = np.moveaxis(resized_image, 0, 1).shape
print(swapped.shape)
tensor = torch.from_numpy(swapped)
print(tensor.shape)
print("####################### Predicted features #######################")
print(features(tensor))
