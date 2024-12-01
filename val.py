from ultralytics import YOLO
import wandb
# from wandb.integration.ultralytics import add_wandb_callback
import torch



# Load a model
model = YOLO("/mnt/data/weights/base_weight/weights/best_wo_specialised_training.pt")


print("Updatedmodelnames>>>>",model.names)
validation_results = model.val(data="/mnt/data/dataset/YOLODataset/dataset.yaml", imgsz=2480, 
                               batch=1, conf=0.15, iou=0.25, 
                               device="0", plots = True,
                               save_json= True
                               )



