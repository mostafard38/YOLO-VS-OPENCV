from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def export_callback_example(exporter):
    print(
        "\n...Callback logs...\n"
        "model export started...\n"
        f"args: {exporter.args}\n"
    )


model.add_callback("on_export_start", export_callback_example)

model.export(format="torchscript")