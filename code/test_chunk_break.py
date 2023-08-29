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
PROMPT = "Rewrite the following dialogue, so that the teacher ask questions deeper and deeper, help to guide the student to learn the knowledge. Dialogue: "
model_save_path = "/cluster/scratch/wangjun/dialog_rewritting/model/model_2/"
base_model = 'google/flan-t5-xl'
# model = AutoModelForSeq2SeqLM.from_pretrained(base_model,  device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(base_model)
prompt_length = len(tokenizer.tokenize(PROMPT))
def break_into_chunks(text, max_effective_length=510):
    prompt_length = len(tokenizer.tokenize(PROMPT))
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





def generate_dialogue(input_text):
    chunks = break_into_chunks(input_text)
    combined_output = ""
    
    for chunk in chunks:
        input_with_prompt = PROMPT + chunk
        print('new one')
        print(input_with_prompt)
        
        input_encoding = tokenizer(input_with_prompt, return_tensors="pt", padding=True)
        
        # Explicitly check token counts
        total_tokens = len(input_encoding["input_ids"][0])
        print('total_tokens',total_tokens)
        if total_tokens > 512:
            # with open('/cluster/scratch/wangjun/dialog_rewritting/result/error.txt', 'a') as f:
            print(f"Offending chunk with {total_tokens} tokens: {input_with_prompt}\n")

            continue
            # print(f"Offending chunk with {total_tokens} tokens: {input_with_prompt}")
            # continue

        # output_ids = model.generate(input_ids=input_encoding["input_ids"].cuda(), max_length=3000, do_sample=True, top_p=0.9, repetition_penalty=1.5)
        # combined_output += tokenizer.decode(output_ids[0], skip_special_tokens=True) + " "
    
    return None

text = "Teacher:  What is the correct way to draw a vector? Student:  Draw in the x and y components of each vector (including the resultant) with a dashed line. Teacher:  What are the components of a triangle? Student:  Use the equations $$\(A_{x} = A\text{cos}\theta\)$$ and $$\(A_{y} = A\text{sin}\theta\)$$ to find the components. Teacher:  What are the four components of a vector? Student:  In Figure 5.23, these components are $$\(A_{x}\)$$, $$\(A_{y}\)$$, $$\(B_{x}\)$$, and $$\(B_{y}.\)$$ Vector $$\(\mathbf{A}\)$$ makes an angle of $$\(\theta_{A}\)$$ with the x-axis, and vector $$\(\mathbf{B}\)$$ makes and angle of $$\(\theta_{B}\)$$ with its own x-axis (which is slightly above the x-axis used by vector A). Teacher:  Which one is greater, the x-axis angle or the total angle between these two? Student:  Figure 5.23 Teacher:  When you have determined these angles, multiply by the number of x and Student:  To add vectors AAA and B,B,B, first determine the horizontal and vertical components of each vector. Teacher:  In Figure 5.24, the Ax and Bx vectors are added to find what? Student:  These are the dotted vectors Ax,Ax,Ax, AyAyAy ByByBy shown in the image. Teacher:  How do you determine the resultant vector? Student:  Find the x component of the resultant by adding the x component of the vectors          $$\[R_{x} = A_{x} + B_{x}\]$$and find the y component of the resultant (as illustrated in Figure 5.24) by adding the y component of the vectors.$$\[R_{y} = A_{y} + B_{y}\text{.}\] Teacher:  What is the most important thing in the answer? Student:  $$Figure 5.24 Teacher:  How does this relate to how much you are paying for a car? Student:  The vectors AxAxAx and BxBxBx add to give the magnitude of the resultant vector in the horizontal direction, Rx.Rx. Teacher:  Which vector has the greatest value? Student:  Rx. Teacher:  What is the greatest value of the vectors AyAyAy and ByByBy? Student:  Similarly, the vectors AyAyAy and ByByBy add to give the magnitude of the resultant vector in the vertical direction, Ry.Ry. Teacher:  What is the value of the atom? Student:  Ry. Teacher:  What other information does this provide? Student:  Now that we know the components of  R,R,R,  we can find its magnitude and direction. Teacher:  What is the magnitude of the resultant R? Student:  To get the magnitude of the resultant R, use the Pythagorean theorem. Teacher:  How does it work? Student:  $$\[R = \sqrt{R_{x}^{2} + R_{y}^{2}}\]$$"
generate_dialogue(text)