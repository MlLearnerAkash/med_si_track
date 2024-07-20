import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO

# Initialize a Weights & Biases run
wandb.init(project="opervu-190724", job_type="training")

from ultralytics import YOLO

# Load a model
model = YOLO("/root/med_si_track/topervu-190724/train/weights/last.pt")  # load a pretrained model (recommended for training)
add_wandb_callback(model, enable_model_checkpointing=True)

# Train the model with 2 GPUs
results = model.train(project= "topervu-190724",data="coco8.yaml", epochs=1000, imgsz=2480, device=0, batch=3,
                      plots =True, resume = True)

# Finalize the W&B Run
wandb.finish()
