from PIL import Image
from io import BytesIO
import base64

def get_image_url(img : Image):
    img_bin = BytesIO()
    img.save(img_bin, format='JPEG')
    img_b64 = base64.b64encode(img_bin.getvalue()).decode('ascii')
    return 'data:image/jpeg;base64, ' + str(img_b64)