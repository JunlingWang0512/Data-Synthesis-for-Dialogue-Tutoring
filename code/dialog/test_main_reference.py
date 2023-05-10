
from typing import List, Tuple
import itertools
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset, concatenate_datasets
import time
import random
from uuid import uuid4
# from dataclasses import dataclass, field #ok

#__tokenizer这部分要删掉————

tokenizer = AutoTokenizer.from_pretrained(
        'google/flan-t5-base',
        cache_dir='/cluster/scratch/wangjun/dialogue_inpainting5_6_both/cache',
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
count = 0
def predict(test):
    
    global count
    result = count
    count += 1
    for example in test:
        print("test_dataset:"+ str(result),example)
    return ["gen" + str(result)]
    

sentences = [
            "Lesmahagow F.C.",
            "Lesmahagow Football Club is a Scottish football club, based in the town of Lesmahagow, South Lanarkshire.",
            "They were formed in 1885 and play at Craighead Park.",
            "Currently playing in the West of Scotland League Central District First Division.",
            "they wear Red And White Hooped Shirts, White Shorts, Red Socks and away colours are Blue Shirt White Trim, Blue Shorts, Blue Socks white trim.",
            "The club are sponsored by The Black Bull Inn, Lesmahagow."
        ]

document_title = "Lesmahagow F.C."
dialog = []  # Initialize an empty dialog
author_num = []

# Generate dialog inpainting
for idx, sentence in enumerate(sentences):
    if idx == 0:
        test_dataset = generate_partial_dialog([sentence], document_title)
    else:
        test_dataset = generate_partial_dialog(dialog + [sentence], document_title)
    
    results = predict(test_dataset)
    # results = method_definition.postprocess_predictions(results, test_dataset)

    # results_to_keep = []
    # for j in range(int(len(results) / model.config.num_return_sequences)):
    #     lower_bound = j * model.config.num_return_sequences
    #     upper_bound = j * model.config.num_return_sequences + model_args.num_sequences_to_keep
    #     results_to_keep.extend(results[lower_bound:upper_bound])
    # results = results_to_keep

    generated_sentence = results[0]

    dialog.append(generated_sentence)  # Add the generated sentence as the first element
    dialog.append(sentence)  # Add the current input sentence
    author_num.append(0)
    author_num.append(1)

    
    # dialog.append(sentence)  # Add the current input sentence
    # author_num.extend([1, 0])

# Save the targeted content
output_data = {
    "title": document_title,
    "pid": str(uuid4()),
    "passage": " ".join(sentences),
    "sentences": sentences,
    "author_num": author_num,
    "utterances": dialog
}
print(output_data)