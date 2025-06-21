from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML

# Train the model
results = model.train(
    data="./train.yaml",
    epochs=30,
    hsv_s=1.0,
    hsv_h=1.0,
    flipud=0.5,
    fliplr=0.5,
    bgr=0.5,
    imgsz=1080,
    dropout=0.1,
    patience=20,
    optimizer="SGD",
    mosaic=0,
    erasing=0,
)
# optimizer SGD
# mosaic 는 빼는 것이 괜찮아 보임
# degrees 는 상관없을 것 같은데 그러면 scale을 제거해서 해볼 필요가 있음
# dropout 같은 경우에는 확인해봤을 때 데이터 수가 많으면 0.2 적으면 0.1 정도가 적당
# train 400장, val 40장
