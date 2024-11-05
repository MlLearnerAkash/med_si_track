from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("/mnt/data/weights/base_weight/weights/best_wo_specialised_training.pt")

print(model.names)
# Define path to directory containing images and videos for inference
source = "/root/ws/med_si_track/test_data/needle.avi"

# Run inference on the source
results = model(source, conf = 0.005, imgsz = 2480, save_json = True,
                show = True, iou= 0.25,
                device = "0")  # generator of Results objects


# #TODO
# from ultralytics import YOLO
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction
# import cv2
# # Load your YOLOv8 model
# yolo_model = YOLO('/root/med_si_track/opervu-240724-train/train/weights/best.pt')

# # Wrap the YOLOv8 model with SAHI's detection model
# detection_model = AutoDetectionModel.from_pretrained(
#     model_type='yolov8',
#     model_path='/root/med_si_track/opervu-240724-train/train/weights/best.pt',
#     confidence_threshold=0.3,
#     device='cuda:0',
#     image_size=1080
# )

# # Perform object tracking with SAHI
# def get_sliced_prediction_with_tracking(image_path, detection_model):
#     # Read the original image
#     original_image = cv2.imread(image_path)

#     if original_image is None:
#         raise ValueError(f"Unable to read the image at path: {image_path}")

#     # Get sliced predictions
#     result = get_sliced_prediction(
#         image=original_image,
#         detection_model=detection_model,
#         slice_height=200,
#         slice_width=200,
#     )

#     # Run tracking on the original image
#     tracked_objects = yolo_model.track(original_image, verbose=False, classes=0,conf=0.5,persist=True)[-1]
#     # print('tracked_objects:', tracked_objects[0].boxes.xyxy)
#     # Draw bounding boxes on the original image
#     for box in tracked_objects.boxes.xyxy:
#         box = box.cpu().numpy().astype(int)
#         cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

#     # Save the modified image
#     output_path = 'output_3.jpg'
#     cv2.imwrite(output_path, original_image)

#     return tracked_objects, output_path

# # Example usage
# try:
#     tracked_results, output_image_path = get_sliced_prediction_with_tracking('/mnt/sdb/needle_images/27470.jpg', detection_model)
#     if tracked_results:
#         print(f"Bounding boxes drawn on the image. Saved at: {output_image_path}")
#     else:
#         print("No tracked results.")
# except Exception as e:
#     print(f"An error occurred: {e}")