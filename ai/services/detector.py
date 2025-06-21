import yaml
import cv2
import torch
from ultralytics import YOLO


class Detector:
    def __init__(self, weights_path, yaml_path, img_size=640, conf_thres=0.25, iou_thres=0.45):
        with open(yaml_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
            self.class_names = self.yaml_data['names']

        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f'using device: {self.device}')

        self.model = YOLO(weights_path)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f'이미지를 로드할 수 없습니다.: {image_path}')
            return None

        results = self.model.predict(
            img,
            imgsz=self.img_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
        )
        return results

    def visualize(self, image_path, save_path=None):
        results = self.detect(image_path)
        if not results:
            return
        output_img = results[0].plot()
        if save_path:
            cv2.imwrite(save_path, output_img)
            print(f'결과 이미지가 저장되었습니다. : {save_path}')
