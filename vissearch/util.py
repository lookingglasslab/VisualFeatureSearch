from PIL import Image
from io import BytesIO
import base64
from typing import Callable

import numpy as np
import cv2
from vissearch.searchtool import get_crop_rect

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