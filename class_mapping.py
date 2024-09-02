import json

# Path to your JSON file
# input_file_path = 'input_data.json'
# output_file_path = 'updated_data.json'

names= ['sponge', 'obstruction', 'scalpel', 'incision', 'woodspack', 'scissors', 'gauze', 'snare', 'black_suture', 'needle', 'glove', 'vesiloop', 'needle_holder', 'forceps', 'sucker', 'clamp']
names_model = ['needle_holder', 'forceps', 'scalpel', 'vesiloop', 'incision', 'needle', 'scissors', 'sucker', 'glove', 'snare', 'black_suture', 'gauze', 'black suture', 'woodspack', 'sponge', 'clamp', 'obstruction']
# Your category ID mapping dictionary
# category_mapping = {
#     0: 12,
#     1: 13,
#     2: 2,
#     3: 4,
#     4: 'woodspack',
#     5: 'scissors',
#     6: 'gauze',
#     7: 'snare',
#     8: 'black_suture',
#     9: 'needle',
#     10: 'glove',
#     11: 'vesiloop',
#     12: 'needle_holder',
#     13: 'forceps',
#     14: 'sucker',
#     15: 'clamp',
#     16: 'obstruction'
# }

# Load the JSON data from the file
# with open(input_file_path, 'r') as json_file:
#     json_data = json.load(json_file)

# # Update the category_id according to the mapping dictionary
# for item in json_data:
#     category_id = item["category_id"]
#     if category_id in category_mapping:
#         item["category_id"] = category_mapping[category_id]

# # Save the updated JSON data to a new file
# with open(output_file_path, 'w') as json_file:
#     json.dump(json_data, json_file, indent=4)

# print(f"Updated JSON data has been saved to '{output_file_path}'.")
index_mapping = {names_model.index(m_name): names.index(n_name) for m_name in names_model for n_name in names if m_name == n_name}

print(index_mapping)