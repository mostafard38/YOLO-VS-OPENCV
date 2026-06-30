from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.predict(
    source=0,
    show=True,
    imgsz=640,
    conf=0.25,
    verbose=False
)