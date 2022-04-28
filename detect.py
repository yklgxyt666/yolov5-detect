import time
from pathlib import Path

import torch as t
import numpy as np
from numpy import random
from PIL import Image

from models.yolo import Model
from torchvision import transforms
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, intersect_dicts

nc = 30
device = t.device('cpu')

## load model
model = t.load('yolov5.pt')['model']
model.float()
model.eval()
# model = Model(ckpt['model'].yaml, ch=3, nc=nc).to(device)

# exclude = ['anchor']
# state_dict = ckpt['model'].float().state_dict()  # to FP32
# state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
# model.load_state_dict(state_dict, strict=False)  # load

## load data
transform1 = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
])

img = Image.open('images/4.jpg')
img = img.resize((480, 480))
#img.save('data/11.jpg')
if img.mode == 'RGBA':
    img = img.convert('RGB')
imgten = transform1(img)

i = imgten.unsqueeze(0)
print(i.shape)
# detection
with t.no_grad():
    res = model(i)[0]
pred = non_max_suppression(res, 0.45, 0.45)
print(pred)

# postprocess
from PIL import ImageDraw

color = ['red', 'black', 'white', 'green', 'yellow']
class_of_insects = ['clj', 'cljy', 'mpc', 'mpcy', 'rbjjd', 'rbjjdy', 'xtn', 'xtny', 'stn', 'stny', 'smtn', 'smtny', 'llyj', 'llyjy', 'hce', 'hcey',
                    'hblce', 'hblcey', 'ste', 'stey', 'ysze', 'yszey', 'yxze', 'yxzey', 'mgbe', 'mgbey', 'rwwde', 'rwwdey', 'sdfd', 'sdfdy']

a = ImageDraw.ImageDraw(img)
for i, obj in enumerate(pred[0]):
    (x1, y1, x2, y2) = obj[:4]
    a.rectangle(((x1, y1), (x2, y2)), fill=None, outline=color[0], width=5)
    a.text((x1, y1), str(int(obj[5]))+class_of_insects[int(obj[5])], fill=color[1])
    a.text((x2, y2), str(i+1), fill=color[1])
    print((x1, y1), (x2, y2))
print(str(i+1)+" objects founded.")
img.save('new.jpg')
