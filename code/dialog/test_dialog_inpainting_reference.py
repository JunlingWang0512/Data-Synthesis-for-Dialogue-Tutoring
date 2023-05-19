from typing import List, Tuple
import itertools
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset, concatenate_datasets
import time
import random
# from dataclasses import dataclass, field #ok

#__tokenizer这部分要删掉————
tokenizer = AutoTokenizer.from_pretrained(
        'google/flan-t5-base',
        cache_dir='/cluster/scratch/wangjun/dialogue_inpainting5_18_flan_lora_xl/cache',
        use_fast=True,
        revision="main",
        use_auth_token=None,
    )
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
#__tokenizer这部分要删掉————
def _truncate_to_max_length(context):
        context_len = 0
        truncated_context = []
        history_max_utterances = 999
        history_max_tokens = 384

        context = context[-history_max_utterances:]
        for turn in context[::-1]:
            if context_len + len(turn) < history_max_tokens:
                truncated_context.append(turn)
                context_len += len(turn)
            else:
                break

        return truncated_context[::-1]
def _tokenize_with_special_tokens(turn):
        user_token = "<user>"
        system_token ="<system>"
        dialog_act = tokenizer(turn["dialog_act"], add_special_tokens=False)["input_ids"]
        user_tag = user_token if turn["user"] == "user" else system_token
        user_tag = tokenizer(user_tag, add_special_tokens=False)["input_ids"]
        text = tokenizer(turn["text"], add_special_tokens=False)["input_ids"]
        return user_tag + dialog_act + text
def _process_dialog_context(context):
        context = [_tokenize_with_special_tokens(turn) for turn in context]
        context = _truncate_to_max_length(context)
        return list(itertools.chain.from_iterable(context))
def _process_response(response):
        response = tokenizer(response)["input_ids"]
        if len(response) > 512:
            response = response[:511] + [response[-1]]
        return response
def generate_partial_dialog(sentences: List[str], document_title: str) -> Tuple[List[dict], str]:
    sequences, labels = [], []
    
    if len(sentences) % 2 == 0:
        raise ValueError("The input 'sentences' must have an odd number of elements.")

    introduction = f"Hello, I am an automated assistant and can answer questions about {document_title}"
    dialog = [{'dialog_act': '', 'text': introduction, 'user': 'user'}]

    for i, text in enumerate(sentences[:-1]):
        user = 'system' if i % 2 == 0 else 'user'
        dialog.append({'dialog_act': '', 'text': text, 'user': user})

    dialog.append({'dialog_act': '', 'text': '<extra_id_0>', 'user': 'system'})
    dialog.append({'dialog_act': '', 'text': sentences[-1], 'user': 'user'})

    dialog_act = ''
    #__在这里写出完整的partial dialog，除了input_ids__
    unique_id = f"{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    new_dict = {
    'id': [unique_id],
    'context': [dialog],
    'dataset_id':['dialog_inpainting'],
    'dialog_act':[''],
    'knowledge':[[]],
    'response': [''],
    
    }
    #__在这里写出完整的partial dialog，除了input_ids__
    
    
    context = _process_dialog_context(dialog)
    
    dialog_act = tokenizer(dialog_act, add_special_tokens=False)["input_ids"] #tokenizer在main里面应该有
    # label = _process_response('')
    bos_token_needed = tokenizer.bos_token is not None
    full_sequence = [[tokenizer.bos_token_id]] if bos_token_needed else []

    full_sequence += [
        dialog_act,
        context,
        [tokenizer.eos_token_id]
    ]

    full_sequence = list(itertools.chain.from_iterable(full_sequence))

    sequences.append(full_sequence)
    # labels.append(label)
    
    # input_ids = sequences

    return_dict = {
    "input_ids": full_sequence,
    }
    # return return_dict
#到这里
#修改load_dataset 参数，使其load我的dataset
    dataset = Dataset.from_dict(new_dict)
    # updated_dataset = dataset.map(
    #     return_dict, #用到input_ids.
    #     batched=False,
    #     # batch_size=5000,
    #     load_from_cache_file=False
    #     )
    updated_dataset = dataset.map(
    lambda example: return_dict,
    batched=False,
    load_from_cache_file=False
    )
    return updated_dataset
    # for config_name in config_names:
    #         all_datasets.append(load_dataset(
    #                 dataset,
    #                 config_name,
    #                 split=split,
    #                 cache_dir=self.model_args.cache_dir,
    #                 data_files=self.data_args.dataset_data_files,
    #                 # dataset_filter_dict=self.data_args.dataset_filter_dict
    #             )
    #         )
    #     return concatenate_datasets(all_datasets)



#base/get_dataset
# if config_name is None:
#             config_name = self.data_args.dataset_config_name

#         if isinstance(self.data_args.dataset_name, list):
#             all_datasets = []
#             for dataset in self.data_args.dataset_name:
#                 all_datasets.append(self._get_single_dataset(dataset, split, config_name))
#             dataset = concatenate_datasets(all_datasets)
#         else:
#             dataset = self._get_single_dataset(self.data_args.dataset_name, split, config_name)

#         old_eval_column_names = dataset.column_names
        
        
#         processed_features = dataset.map(
#         self.preprocess_features_and_maybe_normalize, #用到input_ids.
#         batched=True,
#         batch_size=5000,
#         load_from_cache_file=False
#         )

#         new_eval_column_names = [col for col in processed_features.column_names if col != "mask_contents"]
#         # new_eval_column_names = [col for col in processed_features.column_names if col != "mask_contents" and col != "knowledge"] #junling modify debug
#         with open('/cluster/scratch/wangjun/temp3/processed_features.txt', 'w') as f:
#             f.write(str(processed_features))
#         with open('/cluster/scratch/wangjun/temp3/processed_features_content.txt', 'w') as f:
#             for example in processed_features:
#                 f.write(str(example) + '\n')


#         # Create a new dataset with the updated 'response' values and the additional columns
#         updated_dataset = Dataset.from_dict({
#             key: processed_features[key] if key != "response" else processed_features["mask_contents"]
#             for key in new_eval_column_names
#         })
        
#         with open('/cluster/scratch/wangjun/temp3/updated_dataset.txt', 'w') as f:
#             f.write(str(updated_dataset))
#         with open('/cluster/scratch/wangjun/temp3/updated_dataset_content.txt', 'w') as f:
#             for example in updated_dataset:
#                 f.write(str(example) + '\n')
        
        
#         return updated_dataset


# Example usage:
# sentences = ["What is a physician's assistant?", "Physician assistants are medical providers.", "What do they do?"]
# document_title = "Physician Assistants"

# updated_dataset = generate_partial_dialog(sentences, document_title)
# print('updated_dataset',updated_dataset)
# for example in updated_dataset:
#     print(str(example))
    
    
sentences = [
        "Lesmahagow F.C."
        ]
        
document_title = "Lesmahagow F.C."

test_dataset = generate_partial_dialog(sentences, document_title)
for example in test_dataset:
    print(str(example))
#                 f.write(str(example) + '\n')
# print('labels',labels)
# print("context:", context)
# print("dialog_act:", dialog_act)

#now we try tokenizer part, in preprocessing/generation...
