# from dialog.main import RunMode, main

# if __name__ == "__main__":
#     main(RunMode.PREDICT)
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
nltk.download('punkt')

#__function for dialog inpainting__
tokenizer = AutoTokenizer.from_pretrained(  #can be merged into main function if needed later
        'google/flan-t5-base',
        cache_dir='/cluster/scratch/wangjun/dialogue_inpainting5_6_both/cache',
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

    introduction = f"Hello, I am an automated assistant and can answer questions about {document_title}"
    dialog = [{'dialog_act': '', 'text': introduction, 'user': 'user'}]

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

#__function for dialog inpainting__


# Load peft config for pre-trained checkpoint etc.
peft_model_id = "/cluster/scratch/wangjun/dialogue_inpainting5_18_flan_lora_xl/work/ukp/huggingface/training/HuggingfaceTrainingJob.wrncuVcHOHOI/output/models/epoch-best"
config = PeftConfig.from_pretrained(peft_model_id)

# load base LLM model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,  device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map='auto')
#  load_in_8bit=True, device_map="auto" #junling modify
model.eval()

print("Peft model loaded")





# Load dataset from the hub and get a sample
# /cluster/scratch/wangjun/local_data/book_dataset_v4/business/business_ethics.json
# /cluster/scratch/wangjun/local_data/book_dataset_v4/math/algebra_and_trigonometry.json
with open('/cluster/scratch/wangjun/local_data/book_dataset_v4/business/business_ethics.json') as f:
    dataset = json.load(f)

count_section = 0
for key in dataset:
    
    if key not in ('book_statistics','chapter_concepts','chapter_questions'):
        count_section += 1
        section = dataset[key]['content']
        count = 0
        result = []
        document_title = str(key)
        for paragraph in section:
            sentences = nltk.sent_tokenize(paragraph)
            if(len(sentences) == 1):
                document_title = str(sentences[0])
                
            elif(len(sentences) > 1):
                # print('document_title',document_title)
                # print('sentences',sentences)
                dialog = []  # Initialize an empty dialog
                author_num = []
                # test_datasets = []

                # Generate dialog inpainting
                for idx, sentence in enumerate(sentences):
                    if idx == 0:
                        test_dataset = generate_partial_dialog([sentence], document_title)
                    else:
                        test_dataset = generate_partial_dialog(dialog + [sentence], document_title)
                    
                    # print(test_dataset['input_ids'][0])
                    input_ids_tensor = torch.tensor(test_dataset['input_ids'][0])
                    # add repetition penalty
                    results = model.generate(input_ids=input_ids_tensor.unsqueeze(0).cuda(),do_sample=True, top_p=0.9, repetition_penalty=1.5)
                    prediction = tokenizer.decode(results[0].detach().cpu().numpy(), skip_special_tokens=True) #peft prediction
                    
                    # process the prediction results
                    prediction = prediction.replace('<user>', '')
                    prediction = prediction.replace('user>', '')
                    prediction = prediction.replace('<user', '')
                    prediction = prediction.strip()

                    # for example in test_dataset:
                    #     test_datasets.append(example)
                    # results = postprocess_predictions(results, test_dataset)

                    # results_to_keep = []
                    # num_return_sequences, num_sequences_to_keep = 1,1
                    # for j in range(int(len(results) / num_return_sequences)):
                    #     lower_bound = j * num_return_sequences
                    #     upper_bound = j * num_return_sequences + num_sequences_to_keep
                    #     results_to_keep.extend(results[lower_bound:upper_bound])
                    # results = results_to_keep
                    # with open('/cluster/scratch/wangjun/temp4/prediction.txt', 'w') as f:
                    #     f.write(str(prediction))
                    # for item in prediction:
                    #     with open('/cluster/scratch/wangjun/temp4/prediction_item.txt', 'a') as f:
                    #         f.write(str(item))
                    
                    generated_sentence = prediction

                    dialog.append(generated_sentence)  # Add the generated sentence as the first element
                    dialog.append(sentence)  # Add the current input sentence
                    author_num.append(0)
                    author_num.append(1)
                            
                # Save the targeted content
                output_data = {
                    "title": document_title,
                    "pid": str(uuid4()),
                    "passage": " ".join(sentences),
                    "sentences": sentences,
                    "author_num": author_num,
                    "utterances": dialog
                }
                with open('/cluster/scratch/wangjun/peft_result_save/repetition_penalty1.5_business_ethics_dialogue_inpainting_results.json', 'a') as f:
                        json.dump(output_data, f, cls=NumpyEncoder)
                # if data_args.prediction_output_file is not None:
                #     with open(data_args.prediction_output_file, 'a') as f:
                #         json.dump(output_data, f, cls=NumpyEncoder)
                    
                    
                    # print('outputs',outputs)
                    # results = trainer.predict(test_dataset)
#                     for example in test_dataset:
#                         test_datasets.append(example)
#                     results = method_definition.postprocess_predictions(results, test_dataset)

#                     results_to_keep = []
#                     for j in range(int(len(results) / model.config.num_return_sequences)):
#                         lower_bound = j * model.config.num_return_sequences
#                         upper_bound = j * model.config.num_return_sequences + model_args.num_sequences_to_keep
#                         results_to_keep.extend(results[lower_bound:upper_bound])
#                     results = results_to_keep

#                     generated_sentence = results[0]

#                     dialog.append(generated_sentence)  # Add the generated sentence as the first element
#                     dialog.append(sentence)  # Add the current input sentence
#                     author_num.append(0)
#                     author_num.append(1)
                    
#                 # Save the targeted content
#                 output_data = {
#                     "title": document_title,
#                     "pid": str(uuid4()),
#                     "passage": " ".join(sentences),
#                     "sentences": sentences,
#                     "author_num": author_num,
#                     "utterances": dialog
#                 }
#                 if data_args.prediction_output_file is not None:
#                     with open(data_args.prediction_output_file, 'a') as f:
#                         json.dump(output_data, f, cls=NumpyEncoder)
# dataset = load_dataset("samsum")
# sample = dataset['test'][randrange(len(dataset["test"]))]

# input_ids = tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids.cuda()
# # with torch.inference_mode():
# outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
# print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")

# print(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")





#         with open('/cluster/scratch/wangjun/local_data/book_dataset_v4/math/algebra_and_trigonometry.json') as f:
#             dataset = json.load(f)

#         count_section = 0
#         for key in dataset:
            
#             if key not in ('book_statistics','chapter_concepts','chapter_questions'):
#                 count_section += 1
#                 # if(count_section == 1 or count_section == 2):
#                 #     continue
#                 # if(count_section == 5):
#                 #     break
#                 # print(dataset[key]['content'])
#                 section = dataset[key]['content']
#                 count = 0
#                 result = []
#                 document_title = str(key)
#                 for paragraph in section:
                    
#                     # if len(paragraph) > 69:
#                         # divide paragraph into sentences, and then generate response for each sentence
#                     sentences = nltk.sent_tokenize(paragraph)
#                     if(len(sentences) == 1):
#                         document_title = str(sentences[0])
                        
#                     elif(len(sentences) > 1):
#                         # print('document_title',document_title)
#                         # print('sentences',sentences)
#                         dialog = []  # Initialize an empty dialog
#                         author_num = []
#                         test_datasets = []

#                         # Generate dialog inpainting
#                         for idx, sentence in enumerate(sentences):
#                             if idx == 0:
#                                 test_dataset = generate_partial_dialog([sentence], document_title)
#                             else:
#                                 test_dataset = generate_partial_dialog(dialog + [sentence], document_title)
                            
#                             results = trainer.predict(test_dataset)
#                             for example in test_dataset:
#                                 test_datasets.append(example)
#                             results = method_definition.postprocess_predictions(results, test_dataset)

#                             results_to_keep = []
#                             for j in range(int(len(results) / model.config.num_return_sequences)):
#                                 lower_bound = j * model.config.num_return_sequences
#                                 upper_bound = j * model.config.num_return_sequences + model_args.num_sequences_to_keep
#                                 results_to_keep.extend(results[lower_bound:upper_bound])
#                             results = results_to_keep

#                             generated_sentence = results[0]

#                             dialog.append(generated_sentence)  # Add the generated sentence as the first element
#                             dialog.append(sentence)  # Add the current input sentence
#                             author_num.append(0)
#                             author_num.append(1)
                            
#                         # Save the targeted content
#                         output_data = {
#                             "title": document_title,
#                             "pid": str(uuid4()),
#                             "passage": " ".join(sentences),
#                             "sentences": sentences,
#                             "author_num": author_num,
#                             "utterances": dialog
#                         }
#                         if data_args.prediction_output_file is not None:
#                             with open(data_args.prediction_output_file, 'a') as f:
#                                 json.dump(output_data, f, cls=NumpyEncoder)