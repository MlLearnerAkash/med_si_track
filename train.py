import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO

# Initialize a Weights & Biases run
wandb.init(project="opervu-220824-needle-training", job_type="training")

from ultralytics import YOLO

# Load a model
model = YOLO("/root/ws/med_si_track/weights_2806/last.pt")  # load a pretrained model (recommended for training)
add_wandb_callback(model, enable_model_checkpointing=False)

# Train the model with 2 GPUs
results = model.train(project= "opervu-220824-needle-training",data="/mnt/data/aug_needle_images/YOLODataset/dataset.yaml", epochs=1000, imgsz=2480, device=0, batch=4,
                      plots =True, resume = False, save_period=20)

# Finalize the W&B Run
wandb.finish()
