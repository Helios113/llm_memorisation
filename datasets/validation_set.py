import json

# Load the JSON data from the file
with open('/nfs-share/pa511/llm_memorisation/datasets/PISTOL/sample_data_2.json', 'r') as file:
    data = json.load(file)

# Calculate the split index
split_index = int(len(data) * 0.1)

# Split the data into validation and test/train sets
validation_set = data[:split_index]
test_train_set = data[split_index:]

# Save the validation set to a new JSON file
with open('/nfs-share/pa511/llm_memorisation/datasets/PISTOL/sample_data_2_val.json', 'w') as file:
    json.dump(validation_set, file, indent=4)

# Save the test/train set to a new JSON file
with open('/nfs-share/pa511/llm_memorisation/datasets/PISTOL/sample_data_2_train.json', 'w') as file:
    json.dump(test_train_set, file, indent=4)

print("Data has been split and saved successfully.")