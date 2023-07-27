# post_processing

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import json
from transformers import pipeline, set_seed
import random

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize your models and move them to the device
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-xl').to(device)

def generate_student_answer(student_answer):
    input_text = f"""Translate this student's formal answer into casual, conversational language, but keep the original meaning the same: "{student_answer}" """
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded = re.sub(r'^\'|\'$', '"', decoded)  # replace leading and trailing single quotes with double quotes
    decoded = re.sub(r'^\"|\"$', '', decoded)  # remove leading and trailing double quotes
    return decoded

# def generate_teacher_question(teacher_question, student_answer):
#     input_text = f"""Assuming you are a teacher asking a question to help a student learn a passage, given the answer of this question from student "{student_answer}", rewrite the following question in a teacher style: "{teacher_question}" """
#     input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
#     outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)             
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     decoded = re.sub(r'^\'|\'$', '"', decoded)  # replace leading and trailing single quotes with double quotes
#     decoded = re.sub(r'^\"|\"$', '', decoded)  # remove leading and trailing double quotes
#     return decoded
def generate_teacher_question(teacher_question, student_answer):
    input_text = f"""Assuming you are a teacher asking a question to help a student learn a passage. The student's response is: "{student_answer}". Rewrite the teacher's question ("{teacher_question}") so it can more effectively lead to this answer."""
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)             
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded = re.sub(r'^\'|\'$', '"', decoded)  # replace leading and trailing single quotes with double quotes
    decoded = re.sub(r'^\"|\"$', '', decoded)  # remove leading and trailing double quotes
    return decoded


def is_example_or_metaphor(content):
    # Implement your logic to detect if the content is an example or metaphor
    # This is a placeholder function and always returns False for now
    return False
def remove_unexpected_chars(text):
    # specify the list of characters to remove
    chars_to_remove = 'â€€â€œ'
    # create a translation table that maps every character in chars_to_remove to None
    trans = str.maketrans('', '', chars_to_remove)
    # apply the translation table to the text
    return text.translate(trans)
def post_process(dialog):
    set_seed(42)

    new_dialog = {'title': dialog['title'], 'pid': dialog['pid'], 'passage': dialog['passage'], 
                  'sentences': dialog['sentences'], 'author_num': [], 'utterances': []}
    
    i = 0
    while i < len(dialog['utterances']):
        turn = dialog['utterances'][i]
        author = dialog['author_num'][i]

        if author == 0:  # If the statement is from a teacher
            # Check if we're not at the end and if the next answer is from student and is not a single number
            if i + 1 < len(dialog['utterances']) and dialog['author_num'][i+1] == 1 and not remove_unexpected_chars(dialog['utterances'][i+1].strip()).isdigit() and not dialog['utterances'][i+1].strip().isdigit():
                # question = generator(f"Rewrite the following one question in a way a teacher would say it: {turn}", max_length=int(len(turn.split())*2), temperature=0.7, do_sample=True)[0]['generated_text']
                # question = question.replace("Rewrite the following one question in a way a teacher would say it: ", "")
                answer = remove_unexpected_chars(dialog['utterances'][i+1].strip())
                question = generate_teacher_question(turn, answer)
                if not question.endswith('?'):
                    question = turn
                # answer = generate_student_answer(answer)
                # answer = generator(f"Rewrite the following one answer in a way a student would say it: {answer}", max_length=int(len(answer.split())*2), temperature=0.7, do_sample=True)[0]['generated_text']
                # answer = answer.replace("Rewrite the following one answer in a way a student would say it: ", "")

                new_dialog['utterances'].append(question)
                new_dialog['author_num'].append(author)
                new_dialog['utterances'].append(answer)
                new_dialog['author_num'].append(1)
                i += 2
            else: 
                i += 2 # Skip the pair of teacher's question and student's number answer
        else:  # If the statement is from a student but not after a teacher's turn, just add it to the new dialog
            print(f"Non-alternating utterance detected at index {i}")
            new_dialog['utterances'].append(turn)
            new_dialog['author_num'].append(author)
            i += 1

    return new_dialog

data = []
processed_data = []  # new list for processed dialogues
decoder = json.JSONDecoder()

with open('/cluster/scratch/wangjun/dialogue_inpainting5_18_flan_lora_xl/work/ukp/huggingface/6_27_algebra_search/HuggingfaceSearchJob.8FtuwoC7t0JX/output/search_output.json', 'r') as f:
    text = f.read()
    while text:
        obj, idx = decoder.raw_decode(text)
        data.append(obj)
        text = text[idx:].lstrip()

for i, dialogue in enumerate(data):
    processed_dialogue = post_process(dialogue)
    processed_data.append(processed_dialogue)  # append processed dialogues to list
    print('dialog generated successfully!!!')
    print(processed_dialogue)
    
    # Every 10 dialogues, write to file and clear the processed_data list
    if (i + 1) % 10 == 0:
        with open('/cluster/scratch/wangjun/dialogue_inpainting5_18_flan_lora_xl/work/ukp/huggingface/6_27_algebra_search/HuggingfaceSearchJob.8FtuwoC7t0JX/output/search_output_post.json', 'a') as f:
            for dialog in processed_data:
                json.dump(dialog, f)
            processed_data = []  # clear the list after writing to file

# After all dialogues have been processed, write any remaining dialogues in the list to the file
if processed_data:
    with open('/cluster/scratch/wangjun/dialogue_inpainting5_18_flan_lora_xl/work/ukp/huggingface/6_27_algebra_search/HuggingfaceSearchJob.8FtuwoC7t0JX/output/search_output_post.json', 'a') as f:
        for dialog in processed_data:
            json.dump(dialog, f)