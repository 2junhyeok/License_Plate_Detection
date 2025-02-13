from ultralytics import YOLO
import os

# Load a model
model = YOLO("/mnt/hdd_6tb/jh2020/yolo11n.pt")

# Train the model
train_results = model.train(
    data="/mnt/hdd_6tb/kmj9425/yolo_1/yolov11.yaml",
    epochs=50,
    imgsz=640, 
    device=0,
)
