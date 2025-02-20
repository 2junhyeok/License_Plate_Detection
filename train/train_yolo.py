from ultralytics import YOLO
import os

# Load a model
model = YOLO("/mnt/hdd_6tb/jh2020/yolo11n.pt")

search_space={
    "lr0": (1e-5, 1e-1),
    "degrees": (0.0, 45.0),
}

# Train the model
train_results = model.tune(
    data="/mnt/hdd_6tb/jh2020/plateYOLO11n.yaml",
    epochs=130,
    iterations=300,
    optimizer="AdamW",
    space = search_space,
    plots=False,
    save=False,
    val=False,
    imgsz=640,
)
