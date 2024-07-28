import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO

# Initialize a Weights & Biases run
wandb.init(project="opervu-240724-training", job_type="training")

from ultralytics import YOLO

# Load a model
model = YOLO("/root/med_si_track/opervu-240724-train/train/weights/last.pt")  # load a pretrained model (recommended for training)
add_wandb_callback(model, enable_model_checkpointing=False)

# Train the model with 2 GPUs
results = model.train(project= "opervu-240724-train",data="coco8.yaml", epochs=1000, imgsz=2480, device=0, batch=4,
                      plots =True, resume = True, save_period=10)

# Finalize the W&B Run
wandb.finish()
