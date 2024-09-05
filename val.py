from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback
import torch
#Initialize a Weight and Biases Run
# wandb.init(project = "opervu-val-24082024", job_type= "validation")


# Load a model
model = YOLO("/root/ws/med_si_track/opervu-310824-needle-sponge-training/train/weights/best.pt")
# print("Original model names:", model.names)
# class_mapping = {0: 'sponge', 1: 'obstruction', 2: 'scalpel', 3: 'incision', 4: 'woodspack', 5: 'scissors', 6: 'gauze', 7: 'snare', 8: 'black_suture', 9: 'needle', 10: 'glove', 11: 'vesiloop', 12: 'needle_holder', 13: 'forceps', 14: 'sucker', 15: 'clamp', 16: 'obstruction'}

# model.model.names = class_mapping


# # Save the updated model
# model.save('save_model.pt')
# model = YOLO("save_model.pt")
print("Updated model names>>>>", model.names)
# Customize validation settings
validation_results = model.val(data="/mnt/data/sponge_needle/YOLODataset_seg/dataset.yaml", imgsz=2480, 
                               batch=3, conf=0.35, iou=0.25, 
                               device="0", plots = True,
                               save_json= True
                               )

# wandb.finish()


