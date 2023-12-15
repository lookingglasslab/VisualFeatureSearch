from PIL import Image
from io import BytesIO
import base64
from typing import Callable

import torch
import numpy as np
import cv2
from .searchtool import get_crop_rect

def image_to_durl(img : Image):
    ''' Converts a PIL Image to a data URL '''
    img_bin = BytesIO()
    img.save(img_bin, format='JPEG')
    img_b64 = base64.b64encode(img_bin.getvalue()).decode('ascii')
    return 'data:image/jpeg;base64, ' + str(img_b64)

def durl_to_image(url : str):
    ''' Converts a data URL to a PIL Image '''
    data_b64 = url.split(',')[1]
    data = base64.b64decode(data_b64)
    img_bin = BytesIO(data)
    return Image.open(img_bin)

def create_callback(name : str, func : Callable):
    ''' Required to create a JS-Python Callback in Google Colab '''
    try:
        from google.colab import output
        output.register_callback(name, func)
    except:
        pass

def crop_mask(mask):
  ''' crop all zero-valued rows/cols on the outside of the image '''
  top, left, bot, right = get_crop_rect(mask)
  return mask[top:bot, left:right]

def mask_overlay(image: Image, 
                 x: int, y: int, 
                 mask_size: int, mask: np.ndarray,
                 alpha: int = 0.7, beta: int = 0.4) -> np.ndarray:
    '''add a `mask` over a given `image`.
    
       The image result is `image * alpha + mask * beta`'''

    img = np.asarray(image, dtype=np.float32)
    img /= 256
    img *= alpha

    full_mask = np.zeros((mask_size,mask_size))
    full_mask[y:y+mask.shape[0], x:x+mask.shape[1]] = mask
    full_mask = cv2.resize(full_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    full_mask *= beta

    if len(img.shape) == 3:
        img[:, :, 0] += full_mask
        img[:, :, 1] += full_mask
        img[:, :, 2] += full_mask
    else:
        img += full_mask

    return np.minimum(img, 1)

class FeatureHook(torch.nn.Module):
    ''' Module to extract internal feature data from a model. 
        
        Inputs: `model` is the model itself (e.g. `models.resnet50()`), while `layer` 
        is the layer within to extract features from (e.g. `model.layer4[2].conv2`).
         
        `feature_hook.forward(x)` returns the output of the given `layer` when `x` is inputted into the model.'''
        
    def __init__(self, model, layer):
        super().__init__()
        self.model = model
        layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, layer, input, output):
        self.latest_output = output

    def forward(self, x):
        self.model(x)
        return self.latest_output