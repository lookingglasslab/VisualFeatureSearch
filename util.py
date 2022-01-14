from PIL import Image
from io import BytesIO
import base64
from typing import Callable

# converts a PIL image to a data URL
def get_image_url(img : Image):
    img_bin = BytesIO()
    img.save(img_bin, format='JPEG')
    img_b64 = base64.b64encode(img_bin.getvalue()).decode('ascii')
    return 'data:image/jpeg;base64, ' + str(img_b64)

# sets up a callback function in a Colab environment
def create_callback(name : str, func : Callable):
    try:
        from google.colab import output
        output.register_callback(name, func)
    except:
        pass