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
# # read the JSON file
with open('/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/dialog/datasets/local_data/OR_QUAC/train_filtered.json', 'r') as f:
    data = json.load(f)

# print the length of the data
# print(len(data)) # test 5571 train 7495 validation:3430

# # # Print the first 5 samples
# for sample in data[:2]:
#     print(json.dumps(sample, indent=2))
count = 0
for sample in data:
    # print(sample)
    if 'answer' in sample:
        # count += 1
        if sample["answer"]['text'].strip().lower() == 'cannotanswer':
            count += 1

print(f'Number of samples with "CANNOTANSWER": {count}')

#filter data with "cannotanswer"
# import json

# # Read the JSON file
# with open('/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/dialog/datasets/local_data/OR_QUAC/test.json', 'r') as f:
#     data = json.load(f)

# # Filter samples without "cannotanswer"
# filtered_data = [sample for sample in data if 'answer' not in sample or sample["answer"]['text'].strip().lower() != 'cannotanswer']

# # Save the filtered data to a new JSON file
# with open('/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/dialog/datasets/local_data/OR_QUAC/test_filtered.json', 'w') as f:
#     json.dump(filtered_data, f, indent=2)

# # Print the number of samples removed
# print(f'Number of samples removed: {len(data) - len(filtered_data)}')
