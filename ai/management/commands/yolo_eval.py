from django.core.management.base import BaseCommand
from django.conf import settings
import argparse
import os
from ai.utils.claude_eval import YOLOv4Evaluator
import xml.etree.ElementTree as ET


class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        config = settings.YOLOV4_CONFIG
        weights = settings.YOLOV4_WEIGHTS
        classes = settings.YOLOV4_CLASSES
        eval_data_path = settings.EVAL_DATA_PATH

        # 평가기 초기화
        evaluator = YOLOv4Evaluator(
            config_path=config,
            weights_path=weights,
            class_names_path=classes,
            conf_threshold=0.5,
            nms_threshold=0.4,
        )

        # 평가 수행 (YOLO 형식)
        results = evaluator.evaluate_dataset(
            test_data_path=eval_data_path, iou_threshold=0.3
        )

        # 결과 출력 및 저장
        evaluator.print_evaluation_results(results)
        evaluator.save_results(results, "evaluation_results.json")
