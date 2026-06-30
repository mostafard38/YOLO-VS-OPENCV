
## Training the model 

from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def train_callback_example(trainer):
    print(
        "\n...Callback logs...\n"
        "model training started...\n"
        f"epochs: {trainer.epochs}\n"
        f"batch size: {trainer.batch_size}\n"
        f"device: {trainer.device}\n"
    )


model.add_callback("on_train_start", train_callback_example)

model.train(data="coco8.yaml", batch=8, workers=1, epochs=1)