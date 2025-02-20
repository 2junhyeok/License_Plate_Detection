from ultralytics import YOLO
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Load a model
model = YOLO("/mnt/hdd_6tb/jh2020/config/yolo11n.pt")

# Train the model
train_results = model.train(
    data="/mnt/hdd_6tb/jh2020/processed_plate/data.yaml",
    epochs=100,
    optimizer="AdamW",
    plots=False,
    save=False,
    imgsz=640,
)
