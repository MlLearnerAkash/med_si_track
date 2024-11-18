
import cv2
import numpy as np
import os

def denormalize_points(points, img_width, img_height):
    """
    Convert normalized YOLO polygon points to absolute pixel values.
    
    Parameters:
        points (list of tuples): List of (x, y) normalized coordinates.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        list of tuples: List of (x, y) pixel coordinates.
    """
    return [(int(x * img_width), int(y * img_height)) for x, y in points]

def draw_polygon_annotations(image_path, label_path, output_path):
    """
    Draw YOLO polygonal annotations on a single image and save the result.

    Parameters:
        image_path (str): Path to the input image.
        label_path (str): Path to the YOLO-format annotation file.
        output_path (str): Path to save the annotated image.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}.")
        return

    img_height, img_width = image.shape[:2]

    # Check if label file exists
    if not os.path.exists(label_path):
        print(f"Error: Label file {label_path} not found.")
        return

    # Read annotations
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        if len(data) < 3:
            continue

        class_id = int(data[0])  # Class ID
        points = list(map(float, data[1:]))
        # Convert to (x, y) pairs
        points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
        # Denormalize points to pixel values
        abs_points = denormalize_points(points, img_width, img_height)

        # Draw polygon
        cv2.polylines(image, [np.array(abs_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=6)

        # Optional: Draw class ID near the first point
        cv2.putText(image, str(class_id), abs_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Save the annotated image
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to {output_path}")

if __name__ == "__main__":
    # Input paths for the image and the annotation file
    image_path = "/root/ws/med_si_track/LabelMeToYoloSegmentation/images/train/31883_aug_5.png"       # Replace with your image path
    label_path = "/root/ws/med_si_track/LabelMeToYoloSegmentation/labels/train/31883_aug_5.txt" # Replace with your annotation path
    output_path = "yolo_vis.jpg"    # Replace with your output path

    draw_polygon_annotations(image_path, label_path, output_path)
