from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

#Initialize a Weight and Biases Run
wandb.init(project = "opervu-val-24072024", job_type= "validation")


# Load a model
model = YOLO("/root/med_si_track/opervu-240724-train/train/weights/best.pt")
add_wandb_callback(model, enable_model_checkpointing=False)


# Customize validation settings
validation_results = model.val(data="coco8.yaml", imgsz=2480, 
                               batch=1, conf=0.25, iou=0.6, 
                               device="0", plots = True,
                               )

wandb.finish()