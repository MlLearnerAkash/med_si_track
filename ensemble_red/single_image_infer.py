from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean 

# from map_label import map_classes, map_model_output
# Load your trained models

# print("Model names>>base_model>>>", results_base)
# print(results_needle_2[0].boxes.conf.cpu().numpy())

def extract_labels_and_scores(results):
    labels = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    coord_xyxy = results[0].boxes.xyxy.cpu().numpy()
    
    # Create a dictionary with coordinates as keys and label and score as values
    result_dict = {}
    for i, coord in enumerate(coord_xyxy):
        # Convert the numpy array `coord` to a tuple so it can be used as a key
        result_dict[tuple(coord)] = {"label": labels[i], "score": scores[i]}
    
    return result_dict

import numpy as np

# Function to calculate IoU between two boxes in xyxy format
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # Calculate the coordinates of the intersection rectangle
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    # Calculate area of intersection
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate areas of each bounding box
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    # Calculate union area
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    if union_area == 0:
        return 0
    else:
        return inter_area / union_area

# Function to merge two dictionaries based on overlapping bounding boxes
def merge_predictions(dict1, dict2, iou_threshold=0.5):
    merged_dict = {}

    # Iterate through the first dictionary
    for coord1, info1 in dict1.items():
        merged = False
        # Iterate through the second dictionary
        for coord2, info2 in dict2.items():
            iou = calculate_iou(coord1, coord2)
            if iou > iou_threshold:
                # Merge the labels and scores for overlapping coordinates
                merged_dict[coord1] = {
                    'label': (info1['label'], info2['label']),
                    'score': (info1['score'], info2['score'])
                }
                merged = True
                break
        if not merged:
            # If no overlap, add the original bounding box from dict1
            merged_dict[coord1] = info1

    # Add non-overlapping bounding boxes from dict2
    for coord2, info2 in dict2.items():
        if coord2 not in merged_dict:
            merged_dict[coord2] = info2

    return merged_dict

# Function to draw bounding boxes on an image
def annotate_image(image, boxes, color, label="Before Merge", thickness=5):
    # Iterate over all bounding boxes
    for coord, info in boxes.items():
        x1, y1, x2, y2 = coord
        # Draw rectangle for bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        # Put the label and score on the top of the bounding box
        print("the albel is:::", info['label'], info['score'])
        # text = # Updated label and score handling for tuples
        text = f"{info['label'][0] if isinstance(info['label'], tuple) else info['label']}:{max(info['score']) if isinstance(info['score'], tuple) else info['score']:.2f}"

        cv2.putText(image, text, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image

# Function to visualize the bounding boxes before and after merging
def visualize_before_after_merge(image_path, boxes_before, boxes_after):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    
    # Create copies of the image for annotations
    image_before = image.copy()
    image_after = image.copy()

    # Annotate the image with bounding boxes before merging
    image_before = annotate_image(image_before, boxes_before, color=(0, 255, 0), label="Before Merge")
    
    # Annotate the image with bounding boxes after merging
    image_after = annotate_image(image_after, boxes_after, color=(0, 0, 255), label="After Merge")
    
    # Combine the before and after images side by side for comparison
    combined_image = np.hstack((image_before, image_after))

    # Display the result using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.title("Before Merge (Green) vs After Merge (Red)")
    plt.axis("off")
    plt.savefig("./annotate_image.png")


# Function to find differences between bounding boxes
def find_differences(boxes_before, boxes_after, iou_threshold=0.95):
    different_boxes = {}
    
    # Compare each box in boxes_before with each box in boxes_after
    for coord1, info1 in boxes_before.items():
        found_similar = False
        for coord2, info2 in boxes_after.items():
            iou = calculate_iou(coord1, coord2)
            if iou > iou_threshold:
                found_similar = True
                break
        if not found_similar:
            # If no similar box found, add to different_boxes
            different_boxes[coord1] = info1
    
    # Also check the reverse: boxes in boxes_after that are not in boxes_before
    for coord2, info2 in boxes_after.items():
        found_similar = False
        for coord1, info1 in boxes_before.items():
            iou = calculate_iou(coord1, coord2)
            if iou > iou_threshold:
                found_similar = True
                break
        if not found_similar:
            # If no similar box found, add to different_boxes
            different_boxes[coord2] = info2

    return different_boxes


# Function to visualize the bounding boxes where there's a difference
def visualize_differences(image_path, boxes_before, boxes_after, iou_threshold=0.5):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    
    # Create a copy of the image for annotations
    image_diff = image.copy()

    # Find the bounding boxes that are different between boxes_before and boxes_after
    different_boxes = find_differences(boxes_before, boxes_after, iou_threshold)

    # Annotate the different boxes (both before and after merge)
    image_diff = annotate_image(image_diff, different_boxes, color=(0, 0, 255), label="Difference", thickness=3)
    
    # Display the result using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_diff, cv2.COLOR_BGR2RGB))
    plt.title("Differences (Red)")
    plt.axis("off")
    plt.savefig("./differences.png")
if __name__ == "__main__":

    input_image = "/mnt/data/test_data/frame4.png"

    model_base = YOLO('/root/ws/med_si_track/weights_2806/best.pt')
    model_needle_2 = YOLO('/root/ws/med_si_track/opervu-v8-140924-only-needle-training/train/weights/best.pt')


    results_base = model_base.predict(input_image, save = False, imgsz = 2048, conf = 0.8)
    results_needle_2 = model_needle_2.predict(input_image,save = False, imgsz = 2048, iou= 0.5)
    
    base_pred = extract_labels_and_scores(results_base)
    needle_pred = extract_labels_and_scores(results_needle_2)

    merged_pred = merge_predictions(base_pred, needle_pred)

    print(base_pred)
    print(merged_pred)

    image_path = input_image  # Replace with your image path

    visualize_before_after_merge(image_path, base_pred, needle_pred)
    visualize_differences(image_path, base_pred, needle_pred)