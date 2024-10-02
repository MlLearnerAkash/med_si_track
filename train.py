import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
from ultralytics import RTDETR
# Initialize a Weights & Biases run
wandb.init(project="opervu-011024-needle-only-training", job_type="training")

from ultralytics import YOLO

# Load a model
model = YOLO("/mnt/data/weights/base_weight/weights/best.pt")  # load a pretrained model (recommended for training)
add_wandb_callback(model, enable_model_checkpointing=False)

# Train the model with 2 GPUsd
results = model.train(project= "/mnt/data/weights/opervu-011024-needle-only-training",
                      data="/mnt/data/needle_images_only/needle_images/YOLODataset_seg/dataset.yaml", 
                      epochs=1000, imgsz=2480, device=0, batch=4,
                      plots =True, resume = False, save_period=100)

# Finalize the W&B Run
wandb.finish()
