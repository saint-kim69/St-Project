import cv2
from insightface.model_zoo import get_model

# ArcFace model initialization
arcface = get_model("arcface_r100_v1")
arcface.prepare(ctx_id=0)


def embedding(image_path=None, image=None):
    """Return an ArcFace embedding for the given image."""
    if image_path:
        image = cv2.imread(image_path)
    if image is None:
        return None
    img = cv2.resize(image, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return arcface.get(img).flatten()


class ImageEncoder:
    """Callable class that returns ArcFace embeddings."""

    def __init__(self):
        self.size = (112, 112)

    def __call__(self, image):
        img = cv2.resize(image, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return arcface.get(img).flatten()
