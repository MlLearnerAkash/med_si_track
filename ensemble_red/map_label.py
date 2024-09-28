def map_classes(model_base, model_needle):
    # Reverse the dictionaries to map from label names to indices
    reverse_model_base = {v: k for k, v in model_base.items()}
    reverse_model_needle = {v: k for k, v in model_needle.items()}
    
    # Find common class names between the two models
    common_classes = set(model_base.values()) & set(model_needle.values())
    
    # Create a dictionary to map the indices of common classes
    class_mapping = {}
    
    for class_name in common_classes:
        base_index = reverse_model_base[class_name]
        needle_index = reverse_model_needle[class_name]
        class_mapping[base_index] = needle_index
    
    return class_mapping


def map_model_output(model_output, class_mapping):
    # Create a list to store the mapped output
    mapped_output = []
    
    for index in model_output:
        if index in class_mapping.values():  # Check if the index is present in the mapping values
            # Find the corresponding key in class_mapping based on value from model_output
            base_index = [key for key, value in class_mapping.items() if value == index][0]
            mapped_output.append(base_index)
        else:
            # If the index is not found in the mapping, keep it unchanged or handle it appropriately
            mapped_output.append(None)  # You can modify this based on your use case
    
    return mapped_output
# Example dictionaries (Model Base and Model Needle)
model_base = {
    0: 'incision', 1: 'gauze', 2: 'glove', 3: 'black_suture', 4: 'vesiloop', 
    5: 'needle', 6: 'needle_holder', 7: 'sucker', 8: 'clamp', 9: 'forceps', 
    10: 'obstruction', 11: 'woodspack', 12: 'snare', 13: 'sponge', 
    14: 'scissors', 15: 'bovie', 16: 'scalpel'
}

model_needle = {
    0: 'sponge', 1: 'obstruction', 2: 'scalpel', 3: 'incision', 
    4: 'woodspack', 5: 'scissors', 6: 'gauze', 7: 'snare', 
    8: 'black_suture', 9: 'needle', 10: 'glove', 11: 'vesiloop', 
    12: 'needle_holder', 13: 'forceps', 14: 'sucker', 15: 'clamp'
}

# Perform the class mapping
class_mapping = map_classes(model_base, model_needle)

# Print the results
print("Key-to-Key mapping of common classes between Model Base and Model Needle:")
print(class_mapping)
