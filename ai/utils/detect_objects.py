import cv2
import numpy as np
import yaml
from django.conf import settings
import os

from collections import defaultdict
from ai.dtos.detect_out_dto import DetectOutDto, MetaModel, DetectModel

from ultralytics import YOLO


yaml_path = os.path.join(settings.BASE_DIR, "model/set_weights.yaml")
weights_path = os.path.join(settings.BASE_DIR, "model/yolov8.pt")


with open(yaml_path, "r") as f:
    yaml_data = yaml.safe_load(f)
    if "names" in yaml_data:
        classes = yaml_data["names"]
    else:
        classes = ["object"]

# YOLOv8 모델 로드
model = YOLO(weights_path)


def detect_image(image_path=None, img=None, pre_path=None):
    if pre_path is None:
        pre_path = os.path.join(settings.BASE_DIR, "static")

    if img is None:
        img = cv2.imread(image_path)

    results = model.predict(img, conf=0.5, iou=0.4)

    detect_data = defaultdict(list)
    boxes = results[0].boxes
    for cls_id, xyxy, conf in zip(boxes.cls, boxes.xyxy, boxes.conf):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = map(int, xyxy)
        w = x2 - x1
        h = y2 - y1
        label = classes[cls_id] if cls_id < len(classes) else str(cls_id)
        detect_data[label].append([x1, y1, w, h, float(conf)])
    if (
        len(detect_data["face"]) != 1
        or len(detect_data["nose"]) != 1
        or len(detect_data["eye"]) != 2
    ):
        return None

    detect_out_dto = DetectOutDto(
        face=DetectModel(
            img=None,
            meta=MetaModel(
                **{
                    "x": detect_data["face"][0][0],
                    "y": detect_data["face"][0][1],
                    "w": detect_data["face"][0][2],
                    "h": detect_data["face"][0][3],
                    "confidence": detect_data["face"][0][4],
                }
            ),
        ),
        nose=DetectModel(
            img=None,
            meta=MetaModel(
                **{
                    "x": detect_data["nose"][0][0],
                    "y": detect_data["nose"][0][1],
                    "w": detect_data["nose"][0][2],
                    "h": detect_data["nose"][0][3],
                    "confidence": detect_data["nose"][0][4],
                }
            ),
        ),
        eye_0=DetectModel(
            img=None,
            meta=MetaModel(
                **{
                    "x": detect_data["eye"][0][0],
                    "y": detect_data["eye"][0][1],
                    "w": detect_data["eye"][0][2],
                    "h": detect_data["eye"][0][3],
                    "confidence": detect_data["eye"][0][4],
                }
            ),
        ),
        eye_1=DetectModel(
            img=None,
            meta=MetaModel(
                **{
                    "x": detect_data["eye"][1][0],
                    "y": detect_data["eye"][1][1],
                    "w": detect_data["eye"][1][2],
                    "h": detect_data["eye"][1][3],
                    "confidence": detect_data["eye"][1][4],
                }
            ),
        ),
    )

    try:
        for key, value in detect_data.items():
            for i, v in enumerate(value):
                cropped_img = img[v[1] : v[1] + v[3], v[0] : v[0] + v[2]]
                cv2.imwrite(os.path.join(pre_path, f"{key}_{i}.jpg"), cropped_img)
                name = key if key in ["face", "nose"] else f"{key}_{i}"
                setattr(getattr(detect_out_dto, name), "img", cropped_img)
    except:
        return
    # 결과 이미지 저장
    cv2.imwrite(os.path.join(pre_path, "detected_image.jpg"), img)

    return detect_out_dto
