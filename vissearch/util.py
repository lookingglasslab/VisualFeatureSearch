from PIL import Image
from io import BytesIO
import base64
from typing import Callable

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