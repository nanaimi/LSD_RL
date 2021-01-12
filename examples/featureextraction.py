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

feature_extractor = nn.Sequential(*(list(model.features)))

mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.eval()

feature_network = nn.Sequential(*(list(mobilenet.children())[0]))

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# #Fix the parameters of the feature extractor:
# for param in features.parameters():
#     param.requires_grad = False

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
img_decode =  PIL.Image.open(BytesIO(res)).convert('RGB')
print(type(img_decode))



img_tensor = preprocess(img_decode)
print(img_tensor.shape)
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.shape)

print("########################## feature vectors ##########################")
# img_variable = Variable(img_tensor)
fc_out = feature_extractor(img_tensor)
print(fc_out)
print(fc_out.shape)
print(fc_out.detach().numpy())
print(fc_out.detach().numpy().shape)
# Create figure and axes
# fig,ax = plt.subplots(1,4)
