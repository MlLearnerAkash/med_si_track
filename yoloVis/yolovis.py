import cv2
import os

def denormalize_bbox(center_x, center_y, width, height, img_width, img_height):
    """
    Convert YOLO normalized bounding box values to absolute pixel values.

    Parameters:
        center_x, center_y (float): Normalized center of the rectangle.
        width, height (float): Normalized width and height of the rectangle.
        img_width, img_height (int): Dimensions of the image.

    Returns:
        (x1, y1, x2, y2): Absolute coordinates of the top-left and bottom-right corners.
    """
    x1 = int((center_x - width / 2) * img_width)
    y1 = int((center_y - height / 2) * img_height)
    x2 = int((center_x + width / 2) * img_width)
    y2 = int((center_y + height / 2) * img_height)
    return x1, y1, x2, y2

def draw_rect_annotations(image_path, label_path, output_path):
    """
    Draw YOLO rectangular annotations on a single image and save the result.

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
        if len(data) != 5:
            continue

        class_id = int(data[0])  # Class ID
        center_x, center_y, bbox_width, bbox_height = map(float, data[1:])
        x1, y1, x2, y2 = denormalize_bbox(center_x, center_y, bbox_width, bbox_height, img_width, img_height)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Optional: Draw class ID near the top-left corner
        cv2.putText(image, str(class_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Save the annotated image
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to {output_path}")

if __name__ == "__main__":
    # Input paths for the image and the annotation file
    image_path = "/mnt/data/needle_images_only/aug_needle_images/YOLOSegData/images/train/9798_aug_5.png"       # Replace with your image path
    label_path = "/mnt/data/needle_images_only/aug_needle_images/YOLOSegData/labels/train/9798_aug_5.txt" # Replace with your annotation path
    output_path = "output.jpg"    # Replace with your output path

    draw_rect_annotations(image_path, label_path, output_path)
