#dialog rewriting inference
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import peft
from peft import PeftModel, PeftConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
from utils import NumpyEncoder
# Load Data
PROMPT = "Rewrite the following dialogue, so that the teacher ask questions deeper and deeper, help to guide the student to learn the knowledge. Dialogue: "
# PROMPT = "Transform the dialogue provided below by enhancing the teacher's questions. The teacher should ask probing questions that dive deeper, guiding the student towards a clearer understanding of the topic. The revised dialogue should contain the same or a greater number of conversational turns compared to the original, ensuring it flows naturally. Dialogue:"
# PROMPT = "Rewrite the dialogue provided below. The teacher should ask questions deeper and deeper, guiding the student towards a clearer understanding of the topic. The rewritten dialogue should contain the same or a greater number of conversational turns compared to the original dialogue, ensuring it flows naturally. Dialogue:"

model_save_path = "/cluster/scratch/wangjun/dialog_rewritting/model/model_2/"
base_model = 'google/flan-t5-xl'
model = AutoModelForSeq2SeqLM.from_pretrained(base_model,  device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(base_model)
prompt_length = len(tokenizer.tokenize(PROMPT))
# Load the Lora model
model = PeftModel.from_pretrained(model, model_save_path, device_map='auto')

# Inference Function
def generate_dialogue0(model, input_text):
    input_with_prompt = PROMPT + input_text
    input_encoding = tokenizer(input_with_prompt, return_tensors="pt", truncation=False, padding=True)
    output_ids = model.generate(input_ids=input_encoding["input_ids"].cuda(),max_length=3000,do_sample=True, top_p=0.9, repetition_penalty=1.5)  # Use input_ids as named argument
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def break_into_chunks(text, max_effective_length=510):
    # prompt_length = len(tokenizer.tokenize(PROMPT))
    chunks = []

    # Check if a substring has valid tokenized length
    def has_valid_length(substr):
        return len(tokenizer.tokenize(substr)) + prompt_length <= max_effective_length

    def find_last_teacher_index(s):
        # Look for both Teacher and Teachers
        index1 = s.rfind("Teacher:")
        index2 = s.rfind("Teachers:")
        return max(index1, index2)

    while text:
        # Initially set the chunk to the whole remaining text
        current_chunk = text

        # If the chunk is longer than allowed (when tokenized), shrink it until it fits
        while not has_valid_length(current_chunk):
            last_teacher_index = find_last_teacher_index(current_chunk)

            # If we can't find a "Teacher:" or "Teachers:", then split at the last possible point
            if last_teacher_index == -1:
                current_chunk = current_chunk.rsplit(' ', 1)[0]
            else:
                current_chunk = current_chunk[:last_teacher_index].strip()

            # If the length of the chunk becomes too small or a specific threshold is met, this means 
            # we cannot break it further. At this point, you might want to handle this case (e.g., raise an error)
            # or just accept the chunk as is.
            if len(current_chunk) <= 10:
                break

        # Append the valid chunk to the list
        chunks.append(current_chunk)

        # Update the text to process the remaining part
        text = text[len(current_chunk):].strip()

    return chunks

def break_into_tuples(text):
    # Split by the roles and filter out any empty strings
    parts = [part.strip() for part in re.split(r'Teacher:|Student:', text) if part.strip()]

    tuples = []

    # Group sentences into pairs (Teacher-Student pairs)
    for i in range(0, len(parts) - 1, 2):  # Skipping by 2 to group Teacher-Student pairs
        tuple_text = "Teacher: " + parts[i] + " Student: " + parts[i+1]
        tuples.append(tuple_text)

    return tuples

def generate_dialogue(model, input_text):
    # chunks = break_into_chunks(input_text)
    chunks = break_into_tuples(input_text) 
    combined_output = ""
    
    for chunk in chunks:
        input_with_prompt = PROMPT + chunk
        input_encoding = tokenizer(input_with_prompt, return_tensors="pt", padding=True)
        
        # Explicitly check token counts
        total_tokens = len(input_encoding["input_ids"][0])
        if total_tokens > 512:
            with open('/cluster/scratch/wangjun/dialog_rewritting/result/error.txt', 'a') as f:
                f.write(f"Offending chunk with {total_tokens} tokens: {input_with_prompt}\n")

            continue
            # print(f"Offending chunk with {total_tokens} tokens: {input_with_prompt}")
            # continue

        output_ids = model.generate(input_ids=input_encoding["input_ids"].cuda(), max_length=3000, do_sample=True, top_p=0.9, repetition_penalty=1.5)
        combined_output += tokenizer.decode(output_ids[0], skip_special_tokens=True) + " "
    
    return combined_output.strip()


# input_ids_tensor = torch.tensor(test_dataset['input_ids'][0])
# # add repetition penalty
# results = model.generate(input_ids=input_ids_tensor.unsqueeze(0).cuda(),do_sample=True, top_p=0.9, repetition_penalty=1.5)
# List of sample dialogues
# samples = [
#     "Teacher: What's the topic for today? Student: We're discussing AI.",
#     "Teacher: How was the lecture? Student: I found it a bit confusing.",
#     "Teacher: How do you think the principles of physics contribute to the functioning of the technological devices you use daily? Student: Think about all of the technological devices that you use on a regular basis. Teacher: Can you provide some examples of how principles of physics are applied in the functioning of computers, wireless internet, smartphones, tablets, GPS, MP3 players, and satellite radio? Student: Computers, wireless internet, smart phones, tablets, global positioning system (GPS), MP3 players, and satellite radio might come to mind. Teacher: How do these exciting modern technologies, such as levitating trains and microscopic robots, demonstrate the principles of physics in action? Student: Next, think about the most exciting modern technologies that you have heard about in the news, such as trains that levitate above their tracks, invisibility cloaks that bend light around them, and microscopic robots that fight diseased cells in our bodies. Teacher: What do these groundbreaking advancements in technology demonstrate about the importance of understanding the principles of physics? Student: All of these groundbreaking advancements rely on the principles of physics.",
#     "Teacher: What do we usually call scientific ideas and explanations that are true in many, but not all situations in the universe? Student: Scientific ideas and explanations that are true in many, but not all situations in the universe are usually called principles. Teacher: Can you provide another example of a scientific principle that is true in many situations, but not all? Student: An example is Pascal's principle, which explains properties of liquids, but not solids or gases. Teacher: Why is it important to make a careful distinction between scientific laws and principles? Student: However, the distinction between laws and principles is sometimes not carefully made in science."
#     # Add as many samples as you want
# ]


# output_data = []

def split_into_utterances(dialogue_text):
    # Split by the roles and filter out any empty strings
    parts = [part.strip() for part in re.split(r'Teacher:|Student:|Teachers:|Students:', dialogue_text) if part.strip()]
    return parts

def extract_utterances(data):
    utterances = data['utterances']
    dialog = []

    for idx, utterance in enumerate(utterances):
        role = "Teacher: " if idx % 2 == 0 else "Student: "
        dialog.append(f"{role} {utterance.replace(role + ': ', '')}")
    
    return " ".join(dialog)


#####math
# input_file = '/cluster/scratch/wangjun/keyword+post_flan-t5-xl_result/math_search_output_post.json'
# count = 0
# data = []
# with open(input_file, 'r') as f:
#     decoder = json.JSONDecoder()
#     text = f.read()
#     while text:
#         obj, idx = decoder.raw_decode(text)
#         data.append(obj)
#         text = text[idx:].lstrip()
        
# for item in data:
#     dialog = extract_utterances(item)
#     rewritten = generate_dialogue(model, dialog)
    
#     # Convert the rewritten dialogue back to the list format
#     rewritten_utterances = split_into_utterances(rewritten)

#     output_data = {
#         "source": item['utterances'],  # Using the original utterances for source
#         "rewritten": rewritten_utterances,
#         "passage": item['passage']
#         }
#     # output_data.append({
#     #     "source": item['utterances'],  # Using the original utterances for source
#     #     "rewritten": rewritten_utterances
#     # })
#     with open('/cluster/scratch/wangjun/dialog_rewritting/result/rewrite_flant5xl_math.json', 'a') as f:
#         json.dump(output_data, f, cls=NumpyEncoder)
#     count += 1
#     print('count=',count)

#####math
input_file = '/cluster/scratch/wangjun/keyword+post_flan-t5-xl_result/math_search_output_post.json'
count = 0
data = []
with open(input_file, 'r') as f:
    decoder = json.JSONDecoder()
    text = f.read()
    while text:
        obj, idx = decoder.raw_decode(text)
        data.append(obj)
        text = text[idx:].lstrip()
        
for item in data:
    dialog = extract_utterances(item)
    rewritten = generate_dialogue(model, dialog)
    
    # Convert the rewritten dialogue back to the list format
    rewritten_utterances = split_into_utterances(rewritten)

    output_data = {
        "source": item['utterances'],  # Using the original utterances for source
        "rewritten": rewritten_utterances,
        "passage": item['passage']
        }
    # output_data.append({
    #     "source": item['utterances'],  # Using the original utterances for source
    #     "rewritten": rewritten_utterances
    # })
    with open('/cluster/scratch/wangjun/dialog_rewritting/result/model2_rewrite_flant5xl_math_8_20.json', 'a') as f:
        json.dump(output_data, f, cls=NumpyEncoder)
    count += 1
    print('count=',count)

# #####science
# input_file = '/cluster/scratch/wangjun/keyword+post_flan-t5-xl_result/science_search_output_post.json'
# count = 0
# data = []
# with open(input_file, 'r') as f:
#     decoder = json.JSONDecoder()
#     text = f.read()
#     while text:
#         obj, idx = decoder.raw_decode(text)
#         data.append(obj)
#         text = text[idx:].lstrip()
        
# for item in data:
#     dialog = extract_utterances(item)
#     rewritten = generate_dialogue(model, dialog)
    
#     # Convert the rewritten dialogue back to the list format
#     rewritten_utterances = split_into_utterances(rewritten)

#     output_data = {
#         "source": item['utterances'],  # Using the original utterances for source
#         "rewritten": rewritten_utterances,
#         "passage": item['passage']
#         }
#     # output_data.append({
#     #     "source": item['utterances'],  # Using the original utterances for source
#     #     "rewritten": rewritten_utterances
#     # })
#     with open('/cluster/scratch/wangjun/dialog_rewritting/result/rewrite_flant5xl_science.json', 'a') as f:
#         json.dump(output_data, f, cls=NumpyEncoder)
#     count += 1
#     print('count=',count)


# #####social
# input_file = '/cluster/scratch/wangjun/keyword+post_flan-t5-xl_result/social_search_output_post.json'
# count = 0
# data = []
# with open(input_file, 'r') as f:
#     decoder = json.JSONDecoder()
#     text = f.read()
#     while text:
#         obj, idx = decoder.raw_decode(text)
#         data.append(obj)
#         text = text[idx:].lstrip()
        
# for item in data:
#     dialog = extract_utterances(item)
#     rewritten = generate_dialogue(model, dialog)
    
#     # Convert the rewritten dialogue back to the list format
#     rewritten_utterances = split_into_utterances(rewritten)

#     output_data = {
#         "source": item['utterances'],  # Using the original utterances for source
#         "rewritten": rewritten_utterances,
#         "passage": item['passage']
#         }
#     # output_data.append({
#     #     "source": item['utterances'],  # Using the original utterances for source
#     #     "rewritten": rewritten_utterances
#     # })
#     with open('/cluster/scratch/wangjun/dialog_rewritting/result/rewrite_flant5xl_social.json', 'a') as f:
#         json.dump(output_data, f, cls=NumpyEncoder)
#     count += 1
#     print('count=',count)
