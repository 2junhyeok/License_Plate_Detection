from ultralytics import YOLO
import os

# Load a model
model = YOLO("/mnt/hdd_6tb/jh2020/yolo11n.pt")

# Train the model
train_results = model.train(
    data="/mnt/hdd_6tb/jh2020/YOLOv11n_car.yaml",
    epochs=100,
    optimizer="AdamW",
    plots=False,
    save=False,
    imgsz=640,
)
