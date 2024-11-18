import os
import json

def remove_files_without_annotations(json_directory, image_extensions=(".jpg", ".png")):
    files_without_annotations = []

    # Iterate over all files in the directory
    for filename in os.listdir(json_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(json_directory, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Check if the 'shapes' field exists and has annotations
                if 'shapes' not in data or not data['shapes']:
                    files_without_annotations.append(filename)

                    # Delete the JSON file
                    os.remove(file_path)
                    
                    # Attempt to delete the associated image file
                    image_file_found = False
                    for ext in image_extensions:
                        image_path = os.path.join(json_directory, filename.replace(".json", ext))
                        if os.path.exists(image_path):
                            os.remove(image_path)
                            image_file_found = True
                            break
                    
                    # Log if no associated image file was found
                    if not image_file_found:
                        print(f"No associated image file found for {filename}")

    # Output the results
    num_files_without_annotations = len(files_without_annotations)
    print(f"Number of files without annotations removed: {num_files_without_annotations}")

    # Print names of files that were removed
    if files_without_annotations:
        print("Files without annotations removed:")
        for file in files_without_annotations:
            print(file)

# Usage
json_directory = '/mnt/data/needle_images_only/needle_images/'  # Replace with the path to your JSON files
remove_files_without_annotations(json_directory)




