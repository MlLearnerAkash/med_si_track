import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
# Initialize a Weights & Biases run
wandb.init(project="opervu-141124-needle-only-training", job_type="training")

from ultralytics import YOLO

# Load a model
model = YOLO("/root/ws/med_si_track/yolov8l.pt")  # load a pretrained model (recommended for training)
add_wandb_callback(model, enable_model_checkpointing=False)

# Train the model with 2 GPUsd
results = model.train(project= "/mnt/data/weights/aug_data_no_pretrained_weights",
                      name="opervu-141124-needle-only-training",
                      data="/mnt/data/needle_images_only/aug_needle_images/YOLOSegData/dataset.yaml", 
                      epochs=2000, imgsz=2480, device=0, batch=4,
                      plots =True, resume = False, save_period=-1,
                      save= True,
                      pretrained = False,
                      cache = False,
                    
                      )

# Finalize the W&B Run
wandb.finish()



