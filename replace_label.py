import os
import json
import glob

# Directory containing JSON files
directory_path = '/root/med_si_track/dataset/'

# Function to update the label in a single annotation file
def update_label(annotation):
    for shape in annotation['shapes']:
        if shape['label'] == 'needle?':
            shape['label'] = 'needle'
        if shape['label'] == 'needle ?':
            shape['label'] = 'needle'
        if shape['label'] == 'woods pack':
            shape['label'] = 'woodspack'
        if shape['label'] == 'scissor':
            shape['label'] = 'scissors'
        if shape['label'] == 'black suture':
            shape['label'] = 'black_suture'
        if shape['label'] == 'needle holder':
            shape['label'] = 'needle_holder'
    return annotation

# Process all JSON files in the directory
for file_path in glob.glob(os.path.join(directory_path, '*.json')):
    with open(file_path, 'r') as file:
        annotation_content = json.load(file)
        
    updated_annotation = update_label(annotation_content)
    
    with open(file_path, 'w') as file:
        json.dump(updated_annotation, file, indent=4)

    print(f"Updated {file_path}")

# Example directory path usage:
# directory_path = '/path/to/your/json/files'
