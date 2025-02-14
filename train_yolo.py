from ultralytics import YOLO
import os

# Load a model
model = YOLO("/mnt/hdd_6tb/jh2020/yolo11n.pt")

# Train the model
train_results = model.train(
    data="/mnt/hdd_6tb/jh2020/plateYOLO11n.yaml",
    epochs=50,
    imgsz=640, 
    device=0,
)
