import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
from ultralytics import RTDETR
# Initialize a Weights & Biases run
wandb.init(project="opervu-310824-sponge-needle-training", job_type="training")

from ultralytics import YOLO

# Load a model
model = YOLO("/root/ws/med_si_track/opervu-310824-needle-sponge-training/train/weights/last.pt")  # load a pretrained model (recommended for training)
add_wandb_callback(model, enable_model_checkpointing=False)

# Train the model with 2 GPUsd
results = model.train(project= "opervu-310824-needle-sponge-training",data="/mnt/data/sponge_needle/YOLODataset_seg/dataset.yaml", epochs=1000, imgsz=2480, device=0, batch=4,
                      plots =True, resume = True, save_period=100)

# Finalize the W&B Run
wandb.finish()
