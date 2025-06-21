import cv2
import numpy as np
import os
import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import argparse
import xml.etree.ElementTree as ET


class YOLOv4Evaluator:
    def __init__(
        self,
        config_path,
        weights_path,
        class_names_path,
        conf_threshold=0.5,
        nms_threshold=0.4,
        ext=".txt",
    ):
        """
        YOLOv4 모델 평가를 위한 클래스 초기화

        Args:
            config_path: YOLOv4 설정 파일 경로
            weights_path: 학습된 가중치 파일 경로
            class_names_path: 클래스 이름 파일 경로
            conf_threshold: 신뢰도 임계값
            nms_threshold: NMS 임계값
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.ext = ext
        # YOLO 모델 로드
        self.net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.output_layers = output_layers

        # 클래스 이름 로드
        # with open(class_names_path, "r") as f:
        self.classes = ["face", "eye", "nose"]

        # 평가 결과 저장용
        self.results = {
            "predictions": [],
            "ground_truths": [],
            "inference_times": [],
            "class_metrics": {},
        }

    def preprocess_image(self, image_path):
        """이미지 전처리"""
        image = cv2.imread(image_path)
        height, width, channels = image.shape

        # YOLO 입력을 위한 blob 생성
        blob = cv2.dnn.blobFromImage(
            image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )

        return image, blob, height, width

    def detect_objects(self, blob):
        """객체 탐지 수행"""
        start_time = time.time()

        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        inference_time = time.time() - start_time
        self.results["inference_times"].append(inference_time)
        return outputs

    def postprocess_detections(self, outputs, width, height):
        """탐지 결과 후처리"""
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # NMS 적용
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )

        final_boxes = []
        final_confidences = []
        final_class_ids = []

        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])

        return final_boxes, final_confidences, final_class_ids

    def calculate_iou(self, box1, box2):
        """IoU (Intersection over Union) 계산"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 교집합 영역 계산
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area

    def evaluate_single_image(self, image_path, ground_truth, iou_threshold=0.5):
        """단일 이미지에 대한 평가"""
        # 이미지 처리 및 예측
        image, blob, height, width = self.preprocess_image(image_path)
        outputs = self.detect_objects(blob)
        pred_boxes, pred_confs, pred_classes = self.postprocess_detections(
            outputs, width, height
        )

        # 예측 결과 저장
        predictions = []
        for i, (box, conf, cls) in enumerate(zip(pred_boxes, pred_confs, pred_classes)):
            predictions.append(
                {
                    "bbox": box,
                    "confidence": conf,
                    "class": cls,
                    "class_name": self.classes[cls],
                }
            )

        # mAP 계산을 위한 매칭
        matched_gt = set()
        tp = []  # True Positive
        fp = []  # False Positive
        confidences = []

        for pred in predictions:
            pred_box = pred["bbox"]
            pred_class = pred["class"]
            pred_conf = pred["confidence"]

            confidences.append(pred_conf)

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truth):
                if gt["class"] == pred_class and gt_idx not in matched_gt:
                    iou = self.calculate_iou(pred_box, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp.append(1)
                fp.append(0)
                matched_gt.add(best_gt_idx)
            else:
                tp.append(0)
                fp.append(1)

        # 결과 저장
        self.results["predictions"].extend(predictions)

        return {
            "predictions": predictions,
            "tp": tp,
            "fp": fp,
            "confidences": confidences,
            "num_gt": len(ground_truth),
        }

    def calculate_map(self, evaluation_results, num_classes):
        """mAP (mean Average Precision) 계산"""
        class_aps = []

        for class_id in range(num_classes):
            # 해당 클래스의 예측 결과만 필터링
            class_tp = []
            class_fp = []
            class_confidences = []
            total_gt = 0

            for result in evaluation_results:
                for i, pred in enumerate(result["predictions"]):
                    if pred["class"] == class_id:
                        class_tp.append(result["tp"][i])
                        class_fp.append(result["fp"][i])
                        class_confidences.append(result["confidences"][i])

                # Ground truth 개수 계산
                for gt in result.get("ground_truths", []):
                    if gt["class"] == class_id:
                        total_gt += 1

            if len(class_confidences) == 0 or total_gt == 0:
                class_aps.append(0)
                continue

            # 신뢰도 순으로 정렬
            sorted_indices = np.argsort(class_confidences)[::-1]
            class_tp = np.array(class_tp)[sorted_indices]
            class_fp = np.array(class_fp)[sorted_indices]

            # Precision과 Recall 계산
            tp_cumsum = np.cumsum(class_tp)
            fp_cumsum = np.cumsum(class_fp)

            recalls = tp_cumsum / total_gt
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

            # AP 계산 (11-point interpolation)
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11

            class_aps.append(ap)

            # 클래스별 메트릭 저장
            self.results["class_metrics"][self.classes[class_id]] = {
                "AP": ap,
                "total_predictions": len(class_confidences),
                "total_ground_truths": total_gt,
            }

        map_score = np.mean(class_aps)
        return map_score, class_aps

    def load_yolo_annotation_by_xml(self, annotation_path, img_width, img_height):
        ground_truth = []

        if not os.path.exists(annotation_path):
            return ground_truth

        data = ET.parse(annotation_path)
        for i in data.findall("object"):
            class_name = i.find("name").text
            box = i.find("bndbox")
            xmin = float(box.find("xmin").text)
            xmax = float(box.find("xmax").text)
            ymin = float(box.find("ymin").text)
            ymax = float(box.find("ymax").text)
            if "face" in class_name:
                class_id = 0
            elif "eye" in class_name:
                class_id = 1
            else:
                class_id = 2

            width = xmax - xmin
            height = ymax - ymin
            center_x = xmin
            center_y = ymax

            # YOLO 형식 (center_x, center_y, width, height)을
            # 평가 형식 (x, y, width, height)으로 변환

            ground_truth.append(
                {
                    "class": class_id,
                    "bbox": [int(xmin), int(ymin), int(width), int(height)],
                }
            )
        return ground_truth

    def load_yolo_annotation(self, annotation_path, img_width, img_height):
        """YOLO 형식의 텍스트 어노테이션 파일 로드"""
        ground_truth = []

        if not os.path.exists(annotation_path):
            return ground_truth

        with open(annotation_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 정규화된 좌표를 절대 좌표로 변환
            abs_center_x = center_x * img_width
            abs_center_y = center_y * img_height
            abs_width = width * img_width
            abs_height = height * img_height

            # YOLO 형식 (center_x, center_y, width, height)을
            # 평가 형식 (x, y, width, height)으로 변환
            x = int(abs_center_x - abs_width / 2)
            y = int(abs_center_y - abs_height / 2)
            w = int(abs_width)
            h = int(abs_height)

            ground_truth.append({"class": class_id, "bbox": [x, y, w, h]})

        return ground_truth

    def evaluate_dataset(self, test_data_path, iou_threshold=0.5):
        """데이터셋 전체에 대한 평가 (YOLO 형식 어노테이션)"""
        # 이미지 파일 목록 가져오기
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []

        for file in os.listdir(test_data_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)

        evaluation_results = []

        print(f"평가 시작: {len(image_files)} 개 이미지")

        for idx, image_file in enumerate(image_files):
            if idx % 100 == 0:
                print(f"처리 중: {idx}/{len(image_files)}")

            # 이미지 경로와 어노테이션 경로 설정
            image_path = os.path.join(test_data_path, image_file)
            annotation_file = os.path.splitext(image_file)[0] + self.ext
            annotation_path = os.path.join(test_data_path, annotation_file)

            # 이미지 크기 가져오기
            image = cv2.imread(image_path)
            if image is None:
                print(f"경고: 이미지를 로드할 수 없습니다: {image_path}")
                continue

            img_height, img_width = image.shape[:2]

            if self.ext == ".xml":
                ground_truth = self.load_yolo_annotation_by_xml(
                    annotation_path, img_width, img_height
                )
            else:
                # YOLO 어노테이션 로드
                ground_truth = self.load_yolo_annotation(
                    annotation_path, img_width, img_height
                )

            # 평가 수행
            result = self.evaluate_single_image(image_path, ground_truth, iou_threshold)
            result["ground_truths"] = ground_truth
            result["image_file"] = image_file
            evaluation_results.append(result)

        # mAP 계산
        map_score, class_aps = self.calculate_map(evaluation_results, len(self.classes))

        # 전체 통계 계산
        total_predictions = sum(
            len(result["predictions"]) for result in evaluation_results
        )
        total_tp = sum(sum(result["tp"]) for result in evaluation_results)
        total_fp = sum(sum(result["fp"]) for result in evaluation_results)
        total_gt = sum(result["num_gt"] for result in evaluation_results)

        overall_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        )
        overall_recall = total_tp / total_gt if total_gt > 0 else 0
        f1_score = (
            2
            * (overall_precision * overall_recall)
            / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0
            else 0
        )

        # 평균 추론 시간
        avg_inference_time = np.mean(self.results["inference_times"])
        fps = 1.0 / avg_inference_time

        # 결과 정리
        final_results = {
            "mAP": map_score,
            "class_APs": {self.classes[i]: class_aps[i] for i in range(len(class_aps))},
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "f1_score": f1_score,
            "total_predictions": total_predictions,
            "total_ground_truths": total_gt,
            "avg_inference_time": avg_inference_time,
            "fps": fps,
            "class_metrics": self.results["class_metrics"],
            "total_images": len(evaluation_results),
        }

        return final_results

    def print_evaluation_results(self, results):
        """평가 결과 출력"""
        print("\n" + "=" * 50)
        print("YOLOv4 모델 평가 결과")
        print("=" * 50)

        print(f"mAP@0.5: {results['mAP']:.4f}")
        print(f"전체 정밀도 (Precision): {results['overall_precision']:.4f}")
        print(f"전체 재현율 (Recall): {results['overall_recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"평균 추론 시간: {results['avg_inference_time']:.4f}초")
        print(f"FPS: {results['fps']:.2f}")

        print("\n클래스별 AP:")
        print("-" * 30)
        for class_name, ap in results["class_APs"].items():
            print(f"{class_name}: {ap:.4f}")

        print("\n클래스별 상세 정보:")
        print("-" * 40)
        for class_name, metrics in results["class_metrics"].items():
            print(f"{class_name}:")
            print(f"  AP: {metrics['AP']:.4f}")
            print(f"  예측 수: {metrics['total_predictions']}")
            print(f"  실제 수: {metrics['total_ground_truths']}")

    def save_results(self, results, output_path):
        """결과를 JSON 파일로 저장"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n결과가 {output_path}에 저장되었습니다.")
