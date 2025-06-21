import torch
import cv2
import numpy as np
import json
from tqdm import tqdm  # 진행률 표시 (선택 사항)

# ==============================================================================
# 주의: 이 코드는 YOLOv4 평가의 '개념적'인 흐름을 보여줍니다.
# 실제 사용 시 다음 부분을 사용자 환경에 맞춰 수정해야 합니다:
# 1. YOLOv4 모델 로드 및 추론 방식 (model_loader.py, yolo_utils.py 등)
# 2. 데이터셋 로드 방식 (Dataset 클래스)
# 3. NMS (Non-Maximum Suppression) 함수
# 4. 예측 결과 포맷 변환 (COCO JSON 형식에 맞게)
# ==============================================================================

# --- 0. 설정 및 상수 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 608  # YOLOv4의 일반적인 입력 크기 (사용하는 모델에 맞춰 변경)
CONF_THRES = 0.001  # NMS 적용 전, 낮은 신뢰도 예측을 제거하기 위한 임계값
IOU_THRES = 0.6  # NMS를 위한 IoU 임계값
WEIGHTS_PATH = "path/to/your/yolov4.pt"  # <-- 학습된 YOLOv4 모델 가중치 경로로 변경
# DATA_CONFIG_PATH = 'path/to/your/dataset.yaml' # <-- 데이터셋 설정 파일 경로로 변경

# COCO API를 위한 어노테이션 파일 (정답 라벨이 담긴 JSON 파일)
# 평가 데이터셋의 정답 JSON 파일 경로 (COCO 형식)
ANNOTATION_FILE_PATH = (
    "path/to/your/val_annotations.json"  # <-- 평가 데이터셋의 COCO JSON 경로로 변경
)


# COCO 클래스 ID와 이름 매핑 (COCO 데이터셋을 사용하는 경우)
# 사용자 정의 데이터셋이라면 해당 클래스 ID와 이름으로 변경
COCO_CLASSES = [
    "face",
    "eye",
    "nose",
]
# COCO API는 클래스 ID가 1부터 시작하므로 0번 인덱스는 __background__로 비워둡니다.
# 실제 COCO 클래스 ID 매핑은 공식 COCO API를 따릅니다.
# category_id_map = {name: i + 1 for i, name in enumerate(COCO_CLASSES)} # 사용자 정의 클래스라면 이렇게 매핑


# --- 1. 모델 로드 (YOLOv4 구현체에 따라 달라짐) ---
# 이 부분은 사용자님의 YOLOv4 PyTorch 구현체에서 모델을 로드하는 방식으로 대체해야 합니다.
# 예시: models.py 파일에 Model 클래스가 있고, .cfg 파일을 통해 모델 구조를 정의하는 경우
# from models import Darknet # 예시
# model = Darknet('cfg/yolov4.cfg', IMG_SIZE) # 모델 구조 로드 (cfg 파일 경로)
# model.load_weights(WEIGHTS_PATH) # 가중치 로드
# model.fuse() # (선택 사항) 퓨즈 레이어
# model.to(DEVICE)
# model.eval() # 평가 모드로 설정
# print("YOLOv4 model loaded successfully.")


# --- 더미 모델 로드 (실제 코드에서는 위에 주석 처리된 부분을 대체) ---
class DummyYOLOv4Model(torch.nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        # 간단한 더미 백본과 헤드
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(
            32, (num_classes + 5) * 3
        )  # 임의의 출력 크기 (cls + bbox + obj)
        print(f"Dummy YOLOv4 model initialized for {num_classes} classes.")

    def forward(self, x):
        # 입력은 (B, C, H, W) 형태 (예: B, 3, 608, 608)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        output = self.fc(x)
        # 실제 YOLO 모델은 (B, num_anchors * (5 + num_classes), Grid_H, Grid_W) 형태의
        # 그리드 기반 예측을 출력합니다. 이 더미 모델은 단순히 형태만 맞춥니다.
        # 출력 형태를 (B, num_predictions, 5 + num_classes)로 변환한다고 가정
        num_predictions = 100  # 임의의 예측 수
        output = output.view(x.size(0), num_predictions, -1)  # (B, 100, 5+num_classes)
        return [output]  # 리스트 형태로 반환 (YOLO 모델 출력과 유사)


dummy_model = DummyYOLOv4Model(IMG_SIZE, len(COCO_CLASSES)).to(DEVICE)
dummy_model.eval()
# -----------------------------------------------------------


# --- 2. 평가 데이터셋 로드 및 전처리 ---
# 이 부분도 사용 중인 YOLOv4 구현체의 데이터셋 로더에 맞춰야 합니다.
# 일반적으로 PyTorch의 DataLoader를 사용합니다.
# COCO API와 함께 사용하려면 COCO Dataset 클래스를 정의해야 합니다.


# 더미 데이터셋 로더 (실제 데이터 로더로 대체)
class DummyEvalDataset(torch.utils.data.Dataset):
    def __init__(self, num_images, img_size, num_classes):
        self.num_images = num_images
        self.img_size = img_size
        self.num_classes = num_classes
        # COCO 형식의 더미 annotations
        self.dummy_annotations = self._generate_dummy_annotations(num_images, img_size)

    def _generate_dummy_annotations(self, num_images, img_size):
        annotations = {"images": [], "annotations": [], "categories": []}
        for i, name in enumerate(COCO_CLASSES):
            annotations["categories"].append(
                {"id": i + 1, "name": name, "supercategory": "object"}
            )

        anno_id_counter = 1
        for img_id in range(1, num_images + 1):
            annotations["images"].append(
                {
                    "id": img_id,
                    "width": img_size,
                    "height": img_size,
                    "file_name": f"dummy_img_{img_id}.jpg",
                }
            )
            # 각 이미지에 임의의 1~3개 객체 추가
            num_objects = np.random.randint(1, 4)
            for _ in range(num_objects):
                x1 = np.random.randint(0, img_size // 2)
                y1 = np.random.randint(0, img_size // 2)
                w = np.random.randint(50, img_size - x1)
                h = np.random.randint(50, img_size - y1)

                annotations["annotations"].append(
                    {
                        "id": anno_id_counter,
                        "image_id": img_id,
                        "category_id": np.random.randint(
                            1, len(COCO_CLASSES) + 1
                        ),  # 1부터 클래스 개수까지
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                anno_id_counter += 1
        # 임시 json 파일로 저장 (pycocotools가 파일을 기대하므로)
        temp_anno_path = "temp_dummy_annotations.json"
        with open(temp_anno_path, "w") as f:
            json.dump(annotations, f)
        return temp_anno_path

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # 더미 이미지 (실제 이미지 로드 코드로 대체)
        img_array = np.random.rand(self.img_size, self.img_size, 3) * 255
        img_tensor = (
            torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        )  # HWC -> CHW, 0-1 정규화

        # targets: COCO Dataset 클래스처럼 dict 형태를 가정
        # 실제로는 COCO API로 로드된 어노테이션 정보에서 해당 이미지의 타겟을 가져와야 합니다.
        # 여기서는 더미이므로 간단히 이미지 ID만 전달 (mAP 계산 시 image_id 필요)
        targets_dict = {
            "image_id": torch.tensor([idx + 1]),  # 이미지 ID는 1부터 시작한다고 가정
            "boxes": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ),  # 더미 박스
            "labels": torch.tensor([0], dtype=torch.int64),  # 더미 라벨 (배경)
        }
        return (
            img_tensor,
            targets_dict,
            f"dummy_img_{idx+1}.jpg",
            None,
        )  # img, targets, path, shape


# 더미 데이터셋 및 DataLoader 인스턴스 생성
NUM_EVAL_IMAGES = 100  # 평가할 이미지 수 (실제는 수천장 이상)
eval_dataset = DummyEvalDataset(NUM_EVAL_IMAGES, IMG_SIZE, len(COCO_CLASSES))
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=16,  # 배치 사이즈
    num_workers=4,  # 데이터 로딩 워커 수
    shuffle=False,  # 평가 시에는 셔플 안 함
    pin_memory=True,
    # collate_fn: 객체 탐지 데이터 로딩에 필요한 커스텀 collate_fn (사용자 구현체에 따라 다름)
    # from utils.datasets import collate_fn as default_collate_fn
    # collate_fn=default_collate_fn
)
# -------------------------------------------------------


# --- 3. 추론 및 예측 결과 수집 ---
def run_evaluation(
    model, dataloader, device, conf_thres, iou_thres, num_classes, coco_classes_names
):
    model.eval()  # 모델을 평가 모드로 설정

    # mAP 계산을 위한 COCO 형식 결과 저장 리스트
    # 각 딕셔너리 형태: {"image_id": int, "category_id": int, "bbox": list, "score": float}
    results_for_coco_eval = []

    # 이미지 ID와 파일 경로 매핑 (mAP 계산 시 필요)
    image_id_to_filepath = {}

    print(f"\nStarting evaluation on {len(dataloader.dataset)} images...")
    with torch.no_grad():  # 추론 시에는 기울기 계산 비활성화
        for batch_idx, (imgs, targets, paths, shapes) in enumerate(
            tqdm(dataloader, desc="Evaluating")
        ):
            imgs = imgs.to(
                device
            ).float()  # / 255.0 (이미 DataLoader에서 했으므로 생략)

            # --- YOLOv4 모델 추론 ---
            # model(imgs)의 출력은 YOLOv4 구현체마다 다를 수 있습니다.
            # 보통 (batch_size, num_predictions, 5 + num_classes) 형태의 원시 예측 텐서입니다.
            preds = model(imgs)[0]  # 더미 모델은 [output] 형태로 반환하므로 [0]

            # --- NMS (Non-Maximum Suppression) 적용 ---
            # 이 NMS 함수는 사용하시는 YOLOv4 구현체에 포함된 것을 사용해야 합니다.
            # 없으면 직접 구현해야 합니다. (예: torchvision.ops.nms)
            # from utils.general import non_max_suppression # 예시

            # 더미 NMS (실제 NMS 함수로 대체)
            def dummy_non_max_suppression(
                prediction, conf_thres, iou_thres, classes=None, agnostic=False
            ):
                # prediction: (num_predictions, 5 + num_classes)
                # 더미이므로 단순히 신뢰도 임계값만 적용하고 박스는 변환하지 않음
                output = []
                for pred_per_image in prediction:
                    high_conf = (
                        pred_per_image[..., 4] > conf_thres
                    )  # obj_conf (objectness score)
                    pred_per_image = pred_per_image[high_conf]

                    if pred_per_image.shape[0] == 0:
                        output.append(None)
                        continue

                    # Dummy: x1, y1, x2, y2, conf, cls_idx
                    # 실제 NMS는 IoU 기반으로 박스를 필터링하고 최종 박스를 반환합니다.
                    # 여기서는 그냥 임의의 박스 좌표로 변환하여 반환

                    # Dummy bbox (xywh) -> xyxy
                    boxes = (
                        pred_per_image[..., :4] * IMG_SIZE
                    )  # 정규화된 값 * 이미지 크기 (절대 좌표)
                    # dummy_boxes = pred_per_image[..., :4].clone() # 박스 좌표는 [0,1] 범위라고 가정
                    # dummy_boxes[..., 0] = dummy_boxes[..., 0] - dummy_boxes[..., 2] / 2 # x_center -> x1
                    # dummy_boxes[..., 1] = dummy_boxes[..., 1] - dummy_boxes[..., 3] / 2 # y_center -> y1
                    # dummy_boxes[..., 2] = dummy_boxes[..., 0] + dummy_boxes[..., 2] # x1 + width -> x2
                    # dummy_boxes[..., 3] = dummy_boxes[..., 1] + dummy_boxes[..., 3] # y1 + height -> y2

                    conf_scores = pred_per_image[..., 4]  # 객체성 점수
                    class_scores = pred_per_image[..., 5:]  # 클래스 확률
                    class_labels = (
                        torch.argmax(class_scores, dim=1) + 1
                    )  # 클래스 ID (COCO는 1부터 시작)

                    # 더미 박스 (xyxy)와 점수, 라벨
                    dummy_boxes = torch.tensor(
                        [[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]]
                    )[: pred_per_image.shape[0]] * (
                        IMG_SIZE / 100
                    )  # 임의의 박스 생성

                    output.append(
                        torch.cat(
                            (
                                dummy_boxes,
                                conf_scores.unsqueeze(1),
                                class_labels.unsqueeze(1).float(),
                            ),
                            dim=1,
                        )
                    )
                return output

            pred = dummy_non_max_suppression(
                preds.cpu(), conf_thres, iou_thres
            )  # CPU로 옮긴 후 NMS (NMS는 보통 CPU에서 더 빠름)
            # -----------------------------------------

            for image_idx, detections in enumerate(pred):  # 각 이미지별 탐지 결과
                if detections is None:
                    continue

                # 이미지 ID 가져오기 (데이터 로더에서 이미지 ID를 제공한다고 가정)
                # targets[image_idx]['image_id']는 list of dict 형태의 targets를 가정하므로
                # targets는 collate_fn으로 처리된 배치 형태일 것임.
                # 더미 데이터로더는 targets_dict를 반환하므로 image_id를 여기서 추출

                # 이미지 ID는 dataloader의 idx와 배치 내 위치로 유추 (실제는 targets에서 가져옴)
                current_image_id = targets["image_id"][image_idx].item()

                # 예측 결과를 COCO JSON 형식에 맞게 변환
                # {"image_id": int, "category_id": int, "bbox": [x, y, width, height], "score": float}
                for (
                    *xyxy,
                    conf,
                    cls,
                ) in (
                    detections.tolist()
                ):  # detections는 [x1, y1, x2, y2, conf, cls] 형태
                    # 바운딩 박스 좌표를 [x, y, width, height] 형태로 변환 (COCO 형식)
                    x1, y1, x2, y2 = xyxy
                    bbox_coco_format = [x1, y1, x2 - x1, y2 - y1]

                    results_for_coco_eval.append(
                        {
                            "image_id": current_image_id,
                            "category_id": int(cls),  # 클래스 ID (COCO는 1부터 시작)
                            "bbox": bbox_coco_format,
                            "score": conf,
                        }
                    )

            # (선택 사항) 진행 상황 출력
            # if batch_idx % 10 == 0:
            #    print(f"Processed batch {batch_idx}/{len(dataloader)}")

    print(f"Collected {len(results_for_coco_eval)} detections for evaluation.")

    # --- 4. mAP (mean Average Precision) 계산 (pycocotools 사용) ---
    # COCO API를 사용하여 mAP를 계산합니다.
    # pycocotools는 JSON 파일을 기반으로 작동하므로, 임시 JSON 파일을 생성합니다.

    # 임시 결과 파일 저장
    temp_results_file = "temp_results.json"
    with open(temp_results_file, "w") as f:
        json.dump(results_for_coco_eval, f)

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # 1. 정답(Ground Truth) 어노테이션 로드
        # 이 파일은 평가 데이터셋의 모든 정답 바운딩 박스 정보를 COCO JSON 형식으로 포함해야 합니다.
        cocoGt = COCO(ANNOTATION_FILE_PATH)

        # 2. 모델 예측 결과 로드
        cocoDt = cocoGt.loadRes(temp_results_file)

        # 3. COCOeval 인스턴스 생성
        # iouType='bbox'는 바운딩 박스에 대한 IoU를 기반으로 평가하겠다는 의미
        cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")

        # 4. 평가 실행
        cocoEval.evaluate()  # 평가 계산
        cocoEval.accumulate()  # 결과 취합
        cocoEval.summarize()  # 요약 출력

        # mAP@0.5, mAP@0.5:0.95 등 주요 지표 출력 (자세한 내용은 cocoEval.stats 확인)
        print("\n--- Evaluation Summary ---")
        print(
            f"mAP@0.5 (IoU=0.50)      : {cocoEval.stats[1]:.3f}"
        )  # COCO mAP 기준 mAP@0.5
        print(
            f"mAP@0.5:0.95 (Avg mAP)  : {cocoEval.stats[0]:.3f}"
        )  # COCO mAP 기준 mAP@0.5:0.95

    except ImportError:
        print("\nError: pycocotools not found. Please install it for mAP calculation.")
        print(
            "pip install pycocotools (Linux/macOS) or pip install pycocotools-windows (Windows)"
        )
    except Exception as e:
        print(f"\nError during COCOeval: {e}")
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_results_file):
            os.remove(temp_results_file)
            print(f"Cleaned up {temp_results_file}")
        if os.path.exists(
            eval_dataset.dummy_annotations
        ):  # 더미 어노테이션 파일도 삭제
            os.remove(eval_dataset.dummy_annotations)


# --- 평가 실행 메인 함수 ---
if __name__ == "__main__":
    # 실제 사용 시, 위에서 정의한 dummy_model, eval_dataloader 대신
    # 사용자 프로젝트의 실제 모델과 데이터 로더를 연결해야 합니다.
    # WEIGHTS_PATH와 ANNOTATION_FILE_PATH를 실제 경로로 반드시 변경하세요.
    print("Running YOLOv4 Evaluation (Conceptual).")
    print("Please ensure your WEIGHTS_PATH and ANNOTATION_FILE_PATH are correct.")
    print(
        "And replace DummyYOLOv4Model/DummyEvalDataset/dummy_non_max_suppression with your actual implementation."
    )

    run_evaluation(
        dummy_model,
        eval_dataloader,
        DEVICE,
        CONF_THRES,
        IOU_THRES,
        len(COCO_CLASSES),
        COCO_CLASSES,
    )
