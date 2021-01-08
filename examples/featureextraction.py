import cv2
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from unrealcv import client
from unrealcv.util import read_npy, read_png

import torch.nn as nn
import torchvision.models as models
import torchsummary as summary


mobilenet = models.mobilenet_v2(pretrained=True)

self.features = nn.Sequential(*(list(mobilenet.children())[0]))
#Fix the parameters of the feature extractor:
for param in self.features.parameters():
    param.requires_grad = False

client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game from http://unrealcv.github.io first.')
else:
    print(help_message)

res = client.request('vget /camera/2/lit png')
img = read_png(res)

print("######################### Shape of image #########################")
print(img.shape)

print("####################### Predicted features #######################")
print(features(img))
