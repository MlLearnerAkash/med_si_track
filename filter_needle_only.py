import os
import json

def filter_shapes(data, label_to_keep="needle"):
    """Filter shapes to keep only the ones with a specific label."""
    # Filter shapes based on the specified label
    filtered_shapes = [
        shape for shape in data.get("shapes", []) if shape["label"] == label_to_keep
    ]
    # Update only the "shapes" key with filtered results
    data["shapes"] = filtered_shapes
    return data

def process_directory(directory_path, label_to_keep="needle"):
    """Process all JSON files in a directory, filtering shapes and saving the changes."""
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            
            # Load JSON data from the file
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            # Filter shapes
            filtered_data = filter_shapes(data, label_to_keep=label_to_keep)
            
            # Save the filtered data back to the same file
            with open(file_path, 'w') as file:
                json.dump(filtered_data, file, indent=4)

# Directory containing the JSON files
directory_path = "/mnt/data/oct_2500/needle_image/"

# Process the directory
process_directory(directory_path)
