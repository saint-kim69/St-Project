import base64
import numpy as np
import cv2

def base64_to_img(base64_img):
    if ',' in base64_img:
        base64_data = base64_img.split(',')[1]
    else:
        base64_data = base64_img
    img_bytes = base64.b64decode(base64_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    