import os
import json
import argparse
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image

# Create a global label dictionary to store label-to-index mapping
label_dict = {"needle":0}

def get_label_index(label):
    """Get index for a label, create new entry if it doesn't exist"""
    if label not in label_dict:
        label_dict[label] = len(label_dict)
    return label_dict[label]


# Helper function to convert polygon to YOLO format
def convert_polygon_to_yolo(points, img_width, img_height):
    """Convert polygon points to YOLO segmentation format.
    Each point is normalized by image dimensions."""
    normalized_points = []
    for x, y in points:
        # Normalize coordinates to [0, 1] range
        x_norm = x / img_width
        y_norm = y / img_height
        normalized_points.extend([x_norm, y_norm])
    
    # Convert all coordinates to strings and join with spaces
    return ' '.join(map(str, normalized_points))

# Helper function to convert bounding box to YOLO format
def convert_bbox_to_yolo(points, img_width, img_height):
    x_min, y_min = points[0]
    x_max, y_max = points[1]

    center_x = (x_min + x_max) / 2 / img_width
    center_y = (y_min + y_max) / 2 / img_height
    bbox_width = (x_max - x_min) / img_width
    bbox_height = (y_max - y_min) / img_height

    return f"{center_x} {center_y} {bbox_width} {bbox_height}"

# Function to convert a single JSON file to YOLO format
def convert_labelme_to_yolo(json_file, output_dir, output_format):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Get the image path from the annotation
    img_path = os.path.join(os.path.dirname(json_file), data['imagePath'])
    
    # Open the image using PIL to get the height and width
    with Image.open(img_path) as img:
        width, height = img.size  # width and height of the image

    filename = Path(data['imagePath']).stem
    height, width = height, width  # data['imageHeight'], data['imageWidth']
    
    yolo_lines = []
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        
        # Get label index instead of using label name
        label_idx = get_label_index(label)


        if output_format == "polygon":
            yolo_line = convert_polygon_to_yolo(points, width, height)
        elif output_format == "bbox":
            yolo_line = convert_bbox_to_yolo(points, width, height)
        
        yolo_lines.append(f"{label_idx} {yolo_line}")

    yolo_file = os.path.join(output_dir, f"{filename}.txt")
    with open(yolo_file, 'w') as f:
        f.write('\n'.join(yolo_lines))

# Modified batch processing function with new directory structure
def batch_convert(json_dir, output_dir, val_size, output_format):
    # Create directory structure
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    for split in ["train", "val"]:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # Get all JSON files and their corresponding image files
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]
    train_files, val_files = train_test_split(json_files, train_size=1-val_size)

    # Process training files
    print("Converting training files...")
    for json_file in tqdm(train_files, desc="Training Files", unit="file"):
        # Convert annotations
        convert_labelme_to_yolo(json_file, os.path.join(labels_dir, "train"), output_format)
    
        # Copy corresponding image
        with open(json_file, 'r') as f:
            data = json.load(f)
        img_path = os.path.join(os.path.dirname(json_file), data['imagePath'])
        if os.path.exists(img_path):
            shutil.copy2(img_path, os.path.join(images_dir, "train"))
    # Process validation files
    print("Converting validation files...")
    for json_file in tqdm(val_files, desc="Validation Files", unit="file"):
        # Convert annotations
        convert_labelme_to_yolo(json_file, os.path.join(labels_dir, "val"), output_format)

        # Copy corresponding image
        with open(json_file, 'r') as f:
            data = json.load(f)
        img_path = os.path.join(os.path.dirname(json_file), data['imagePath'])
        if os.path.exists(img_path):
            shutil.copy2(img_path, os.path.join(images_dir, "val"))

    # Create dataset YAML configuration
    create_dataset_yaml(output_dir)

def create_dataset_yaml(output_dir):
    # Create a list of classes based on label_dict
    classes = [""] * len(label_dict)
    for label, idx in label_dict.items():
        classes[idx] = label
    
    # Create YAML content
    yaml_content = f"""path: {os.path.abspath(output_dir)}
                    train: images/train
                    val: images/val

                    nc: {len(classes)}
                    names: {classes}"""

    # Write YAML file
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Save label dictionary for reference
    dict_path = os.path.join(output_dir, "label_dict.json")
    with open(dict_path, 'w') as f:
        json.dump(label_dict, f, indent=4)
    
    print(f"Created dataset configuration at: {yaml_path}")
    print(f"Saved label dictionary at: {dict_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert LabelMe annotations to YOLO format.")
    parser.add_argument("--json_dir", required=True, help="Directory containing LabelMe JSON files.")
    parser.add_argument("--val_size", type=float, default=0.05, help="Validation size (e.g., 0.1 for 10% validation).")
    parser.add_argument("--output_format", default="polygon", choices=["polygon", "bbox"], help="YOLO output format.")
    parser.add_argument("--output_dir", required=True, help="Directory to save YOLO formatted files.")
    args = parser.parse_args()
    
    batch_convert(args.json_dir, args.output_dir, args.val_size, args.output_format)

if __name__ == "__main__":
    main()
