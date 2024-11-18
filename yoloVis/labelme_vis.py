import json
import cv2
import numpy as np

def draw_annotations(json_path, output_path):
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Read the image
    image = cv2.imread(data['imagePath'])
    
    # Convert points to numpy array for drawing
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            
            # Draw the polygon
            # Use line color if specified, otherwise default to green
            line_color = shape['line_color'] if shape['line_color'] else (0, 255, 0)
            # Use fill color if specified, otherwise default to semi-transparent red
            fill_color = shape['fill_color'] if shape['fill_color'] else (0, 0, 255, 128)
            
            # Draw filled polygon
            overlay = image.copy()
            cv2.fillPoly(overlay, [points], fill_color[:3])  # Fill without alpha
            
            # Apply transparency
            alpha = fill_color[3] / 255 if len(fill_color) == 4 else 0.5
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            # Draw polygon outline
            cv2.polylines(image, [points], True, line_color, 2)
            
    # Save the annotated image
    cv2.imwrite(output_path, image)

# Example usage
json_path = "/mnt/data/needle_images_only/aug_needle_images/31883_aug_5.json"
output_path = "31883_aug_5.png"
draw_annotations(json_path, output_path)
