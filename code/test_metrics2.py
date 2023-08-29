import json
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def calculate_coherence_score(file_path):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")

    decoder = json.JSONDecoder()
    data = []
    with open(file_path, 'r') as f:
        text = f.read()
        while text:
            obj, idx = decoder.raw_decode(text)
            data.append(obj)
            text = text[idx:].lstrip()

    coherence_scores = []

    for obj in data:
        dialog = obj['utterances']
        dialog_str = "\n".join([f"Teacher asks: {dialog[i]}\nStudent answers: {dialog[i+1]}" for i in range(0, len(dialog), 2)])
        
        prompt = f"Assuming you are a linguistic expert, your task is to assess the coherence of the following dialogue generated by an AI system. This dialogue is structured as a teacher-student interaction centered around teaching a passage from a textbook. When we say 'coherence', we're asking: Does the dialogue logically flow and stay consistent with the conversation's context? Does the AI-generated dialogue maintain the thematic context from beginning to end? Please provide a coherence score from 0 to 5, where 0 signifies no coherence and 5 denotes excellent coherence. The dialog is: \n\n{dialog_str}\n\n Please provide me with the score only in your response in this format: Score:<number>"
        print('prompt:',prompt)
        print('__________________________generated______________________________________________')
        inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
        outputs = model.generate(inputs, max_length=512, do_sample=True)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated)
        # Placeholder logic to extract score from generated text
        # This assumes that the model's response is a numerical score, which may not always be the case
        coherence_score = float(generated.split()[-1])
        coherence_scores.append(coherence_score)

    avg_coherence = sum(coherence_scores) / len(coherence_scores)

    df = pd.DataFrame({
        'coherence': [avg_coherence],
    })

    return df

file_path = '/cluster/scratch/wangjun/dialogue_inpainting5_18_flan_lora_xl/work/ukp/huggingface/6_27_algebra_search/HuggingfaceSearchJob.8FtuwoC7t0JX/output/search_output_post.json'
df = calculate_coherence_score(file_path)
print(df)