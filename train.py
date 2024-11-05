import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
from ultralytics import RTDETR
# Initialize a Weights & Biases run
wandb.init(project="opervu-021124-needle-only-training", job_type="training")

from ultralytics import YOLO

# Load a model
model = YOLO("/root/ws/med_si_track/custom_config_needle_modified_head/train/weights/last.pt")  # load a pretrained model (recommended for training)
add_wandb_callback(model, enable_model_checkpointing=False)

# Train the model with 2 GPUsd
results = model.train(project= "./custom_config_needle_modified_head",
                      data="/mnt/data/needle_images_only/needle_images/YOLODataset_seg/dataset.yaml", 
                      epochs=2000, imgsz=2480, device=0, batch=4,
                      plots =False, resume = True, save_period=-1,
                      save= True,
                    #   pretrained = False,
                    
                      )

# Finalize the W&B Run
wandb.finish()



