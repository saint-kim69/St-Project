import cv2
import numpy as np
import yaml
from django.conf import settings
import os

# YAML에서 클래스 이름 읽기 (필요한 경우)
import yaml
from collections import defaultdict
from rest_framework.exceptions import ValidationError
from uuid import uuid4
from ai.dtos.detect_out_dto import DetectOutDto, MetaModel, DetectModel


yaml_path = os.path.join(settings.BASE_DIR, "model/set_weights.yaml")
weights_path = os.path.join(settings.BASE_DIR, "model/yolov4_final.weights")
cfg_path = os.path.join(settings.BASE_DIR, "model/yolov4.cfg")


with open(yaml_path, "r") as f:
    yaml_data = yaml.safe_load(f)
    if "names" in yaml_data:
        classes = yaml_data["names"]
    else:
        # YAML에 클래스 이름이 없는 경우 기본값 설정
        classes = ["object"]

# Darknet 모델 로드
net = cv2.dnn.readNet(weights_path, cfg_path)

# CPU 또는 GPU 선택
# CUDA 사용 가능한 경우 GPU 사용
net.setPreferableBackend(
    cv2.dnn.DNN_BACKEND_CUDA
    if cv2.cuda.getCudaEnabledDeviceCount() > 0
    else cv2.dnn.DNN_BACKEND_OPENCV
)
net.setPreferableTarget(
    cv2.dnn.DNN_TARGET_CUDA
    if cv2.cuda.getCudaEnabledDeviceCount() > 0
    else cv2.dnn.DNN_TARGET_CPU
)

# 출력 레이어 이름 가져오기
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def detect_image(image_path=None, img=None, pre_path=None):
    if pre_path is None:
        pre_path = os.path.join(settings.BASE_DIR, f"static")

    if img is None:
        img = cv2.imread(image_path)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    result = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9:  # 신뢰도 임계값
                # 객체 위치 계산
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 좌표 계산
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    font = cv2.FONT_HERSHEY_SIMPLEX
    detect_data = defaultdict(list)
    clone_img = img.copy()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]

            detect_data[label].append([x, y, w, h, confidence])

            cv2.rectangle(clone_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                clone_img, f"{label} {confidence:.2f}", (x, y - 10), font, 0.5, color, 2
            )
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
    # 결과 그리기
    cv2.imwrite(os.path.join(pre_path, "detected_image.jpg"), img)
    # cv2.imwrite(os.path.join(pre_path, "detected_image_clone.jpg"), clone_img)

    return detect_out_dto
