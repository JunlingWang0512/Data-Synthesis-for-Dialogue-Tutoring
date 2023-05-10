# import json
# import re

# # read the text file
# with open('/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/dialog/datasets/local_data/OR_QUAC/dev.txt', 'r') as f:
#     data = f.read()

# # add newline characters between JSON objects
# data = data.replace('}{', '}\n{')

# # split the text file into individual JSON objects
# json_objects = data.split('\n')

# # clean up any partially formed JSON objects
# json_objects_clean = []
# for obj in json_objects:
#     if obj.strip() == '':
#         continue  # skip the empty line

#     try:
#         json.loads(obj)
#         json_objects_clean.append(obj)
#     except json.JSONDecodeError as e:
#         print(f'Error processing JSON object: {obj}')
#         print(f'Error message: {str(e)}')

#         # Add the missing quote and closing square bracket
#         fixed_obj = re.sub(r'(by B)$', r'by B"]}', obj)

#         try:
#             json.loads(fixed_obj)
#             json_objects_clean.append(fixed_obj)
#             print(f'Successfully fixed JSON object: {fixed_obj}')
#         except json.JSONDecodeError as e:
#             print(f'Failed to fix JSON object: {fixed_obj}')
#             print(f'Error message: {str(e)}')

# # Save the cleaned JSON objects to a new JSON file
# with open('/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/dialog/datasets/local_data/OR_QUAC/validation.json', 'w') as f:
#     json.dump([json.loads(obj) for obj in json_objects_clean], f, indent=4)


import json
# # # # read the JSON file
with open('/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/dialog/datasets/local_data/QRECC/qrecc_train2.json', 'r') as f:
    data = json.load(f)

# # print the length of the data
print(len(data)) # test 16451, train:50800, validation:12701

# # # # Print the first 5 samples
for sample in data[:2]:
    print(sample['Turn_no'] == 1)
    print(json.dumps(sample, indent=2))

# split dataset
# import json
# import random

# # Load the dataset
# with open('qrecc_train.json', 'r') as f:
#     data = json.load(f)

# # Shuffle the data to ensure a random distribution of samples
# random.seed(42)
# random.shuffle(data)

# # Split the dataset into train and validation sets
# train_data = data[:50800]
# validation_data = data[50800:63501]

# # Save the new train dataset
# with open('qrecc_train2.json', 'w') as f:
#     json.dump(train_data, f)

# # Save the validation dataset
# with open('validation.json', 'w') as f:
#     json.dump(validation_data, f)

#new size: qrecc_train2.json :50800
# qrecc_test.json :16451
#validation:12701
# import json

# # Load the dataset
# with open('qrecc_train.json', 'r') as f:
#     data = json.load(f)

# # Initialize containers
# train_data = []
# validation_data = []
# current_data = []

# train_size = 50800
# current_size = 0

# # Iterate through the data and split it based on dialogues
# for sample in data:
#     current_data.append(sample)

#     if sample['Turn_no'] == 1 and len(current_data) > 1:
#         if current_size + len(current_data) - 1 <= train_size:
#             train_data.extend(current_data[:-1])
#             current_size += len(current_data) - 1
#         else:
#             validation_data.extend(current_data[:-1])

#         current_data = [sample]

# # Handle the last dialogue in the dataset
# if current_size + len(current_data) <= train_size:
#     train_data.extend(current_data)
# else:
#     validation_data.extend(current_data)

# # Save the new train dataset
# with open('qrecc_train2.json', 'w') as f:
#     json.dump(train_data, f)

# # Save the validation dataset
# with open('validation.json', 'w') as f:
#     json.dump(validation_data, f)
