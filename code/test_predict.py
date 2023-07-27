import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from random import randrange
from utils import NumpyEncoder
import json
import itertools
from datasets import Dataset, load_dataset, concatenate_datasets
import time
import random
from typing import List, Tuple
import nltk
from uuid import uuid4
import sys
nltk.download('punkt')





tokenizer = AutoTokenizer.from_pretrained(  #can be merged into main function if needed later
        'google/flan-t5-base',
        cache_dir='/cluster/scratch/wangjun/dialogue_inpainting5_18_flan_lora_xl/cache',
        use_fast=True,
        revision="main",
        use_auth_token=None,
    )
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
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
def postprocess_predictions(p, dataset):
        # model_class = self.get_model_class(self.config)

        p.predictions[p.predictions == -100] = tokenizer.pad_token_id
        out = tokenizer.batch_decode(
            p.predictions, skip_special_tokens=True
        )
        return out







def generate_partial_dialog(sentences: List[str], document_title: str) -> Tuple[List[dict], str]:
    sequences, labels = [], []
    
    if len(sentences) % 2 == 0:
        raise ValueError("The input 'sentences' must have an odd number of elements.")

    # introduction = f"Hello, I am an automated assistant and can answer questions about {document_title}"
    # introduction = f"As a teacher, let's start learning about {document_title}."
    # dialog = [{'dialog_act': '', 'text': introduction, 'user': 'system'}] # 'system' acts as teacher

    introduction = f"As your teacher, I'll be asking you many questions about this document. Let's start with this: what's the title of our study material?"
    title_answer = f"The title of our study material is {document_title}"
    dialog = [{'dialog_act': '', 'text': introduction, 'user': 'system'}, {'dialog_act': '', 'text': title_answer, 'user': 'user'}]


    for i, text in enumerate(sentences[:-1]):
        user = 'system' if i % 2 == 0 else 'user'
        dialog.append({'dialog_act': '', 'text': text, 'user': user})

    dialog.append({'dialog_act': '', 'text': '<extra_id_0>', 'user': 'system'})
    dialog.append({'dialog_act': '', 'text': sentences[-1], 'user': 'user'})

    dialog_act = ''
    
    
    unique_id = f"{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    new_dict = {
    'id': [unique_id],
    'context': [dialog],
    'dataset_id':['dialog_inpainting'],
    'dialog_act':[''],
    'knowledge':[[]],
    'response': [''],
    
    }
    
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
    
    dataset = Dataset.from_dict(new_dict)
    

    updated_dataset = dataset.map(
    lambda example: return_dict,
    batched=False,
    load_from_cache_file=False
    )
    return updated_dataset

x = generate_partial_dialog(['sentence1'],"title")
#
for sample in x:
    print(sample)