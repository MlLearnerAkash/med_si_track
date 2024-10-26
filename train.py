# import wandb
# from wandb.integration.ultralytics import add_wandb_callback

# from ultralytics import YOLO
# from ultralytics import RTDETR
# # Initialize a Weights & Biases run
# wandb.init(project="opervu-021024-needle-only-training", job_type="training")

# from ultralytics import YOLO

# # Load a model
# model = YOLO("/root/ws/med_si_track/ultralytics/cfg/models/v8/needle.yaml")  # load a pretrained model (recommended for training)
# add_wandb_callback(model, enable_model_checkpointing=False)

# # Train the model with 2 GPUsd
# results = model.train(project= "./custom_needle",
#                       data="/mnt/data/needle_images_only/needle_images/YOLODataset_seg/dataset.yaml", 
#                       epochs=1, imgsz=480, device=0, batch=4,
#                       plots =False, resume = False, save_period=-1,
#                       save= True,
#                     #   model = "/root/ws/med_si_track/ultralytics/cfg/models/v8/needle.yaml",
#                       pretrained = False,
#                       simplify = True, dynamic = True,
#                       )

# # Finalize the W&B Run
# wandb.finish()



#Custom Needle Training
import wandb
from ultralytics import YOLO

# Initialize a Weights & Biases run
# wandb.init(project="opervu-021024-needle-only-training", job_type="training")

# # Step 1: Train the model if it is not already trained, then save it as .pt
# # If you haven't already trained the model, uncomment the following lines to train it:
# model = YOLO("/root/ws/med_si_track/ultralytics/cfg/models/v8/needle.yaml")
# model.train(data="/mnt/data/needle_images_only/needle_images/YOLODataset_seg/dataset.yaml", epochs=1)
# model.save("needle_large.pt")

# Load the trained .pt model
model = YOLO("/root/ws/med_si_track/runs/detect/train2/weights/best.pt")  # Load the trained model in .pt format

# Add W&B callback (ensure model checkpointing is disabled if not needed)
# add_wandb_callback(model, enable_model_checkpointing=False)

# Train the model
results = model.train(project="custom_needle_large",
                      data="/mnt/data/needle_images_only/needle_images/YOLODataset_seg/dataset.yaml", 
                      epochs=1000, imgsz=2048, device=0, batch=4,
                      plots=False, resume=False, save_period=-1,
                      save=True, pretrained=False, name= "large"
                      )

# Finalize the W&B Run
wandb.finish()
