from ultralytics import YOLO

# Load a model
model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="coco8.yaml", epochs=1000, imgsz=1280, device=0, batch=-1, plots =True, )