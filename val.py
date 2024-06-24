from ultralytics import YOLO

# Load a model
model = YOLO("/workspace/med_si_track/runs/detect/train2/weights/last.pt")

# Customize validation settings
validation_results = model.val(data="coco8.yaml", imgsz=1280, batch=16, conf=0.25, iou=0.6, device="0")