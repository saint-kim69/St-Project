import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import vit_b_16
import numpy as np
from scipy.spatial.distance import cosine
import tensorflow as tf
from django.conf import settings
import os
import cv2

model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

model_path = os.path.join(settings.BASE_DIR, "model/mars-small128.pb")
session = tf.compat.v1.Session()
with tf.io.gfile.GFile(model_path, "rb") as file_handle:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(file_handle.read())
tf.import_graph_def(graph_def, name="net")
input_var = tf.compat.v1.get_default_graph().get_tensor_by_name("%s:0" % "images")
output_var = tf.compat.v1.get_default_graph().get_tensor_by_name("%s:0" % "features")


def embedding(image_path=None, image=None):
    if image_path:
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.fromarray(image)
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(img_tensor)
    return features.flatten()


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


class ImageEncoder:

    def __init__(self):
        self.target_size = (64, 128)
        self.pad_color = (0, 0, 0)

    def __call__(self, image):
        pre_processed_image = self.pre_process_image(image)
        images = np.expand_dims(pre_processed_image, axis=0)
        embedding = session.run(output_var, feed_dict={input_var: images})[0]
        return embedding

    def pre_process_image(self, image):
        return self._add_pad_image(self._resize_image(image))

    def _resize_image(self, image):
        original_height, original_width = image.shape[:2]
        target_width, target_height = self.target_size

        scale = min(
            float(target_width) / original_width, float(target_height) / original_height
        )
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        return cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

    def _add_pad_image(self, image):
        new_height, new_width = image.shape[:2]
        target_width, target_height = self.target_size
        padding_top = (target_height - new_height) // 2
        padding_bottom = target_height - new_height - padding_top
        padding_left = (target_width - new_width) // 2
        padding_right = target_width - new_width - padding_left
        return cv2.copyMakeBorder(
            image,
            padding_top,
            padding_bottom,
            padding_left,
            padding_right,
            cv2.BORDER_CONSTANT,
            value=self.pad_color[::-1],
        )
