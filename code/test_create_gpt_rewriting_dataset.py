import openai
import json
import time
# Step 1: Extract utterances

def extract_utterances(data):
    utterances = data['utterances']
    dialog = []

    for idx, utterance in enumerate(utterances):
        role = "Teacher: " if idx % 2 == 0 else "Student: "
        dialog.append(f"{role} {utterance.replace(role + ': ', '')}")
    
    return " ".join(dialog)

# Step 2: Use GPT-3.5-turbo API for rewriting
def generate_response0(prompt, model):
    messeage_content = prompt
    print(messeage_content)
    
    while True:  # Loop indefinitely until successful
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": messeage_content}]
            )
            return completion
        except:  # Catch any exception
            print("Error occurred while generating response. Retrying in 2 seconds...")
            time.sleep(2)  # Wait for 2 seconds before retrying
def rewrite_dialog(api_key, dialog):
    openai.api_key = api_key

    prompt = (f"Task: rewrite the provided dialog utterance\n"
              "Instructions:\n"
              "1.The current dialog consists of teacher-student exchanges. I hope to make it more coherent"
              "2. Each teacher's question should have good coherence, and there can be more diversity in students' answer."
              "3.The teacher should ask questions deeper and deeper, help to guide the student to learn the knowledge"
              "4.The turn of this dialog is not fixed, you can make more turns but not less."
              "5.Make the overall dialog more coherent and more natural just like human dialog."
              f"6.my current dialog is:\n{dialog}\n"
              "please output dialog in the same format as my input dialog, do not include newline in the output, use normal symbols so I can paste into json file. Use double \ instead of single \ in your output.")
    

    result = generate_response0(prompt,'gpt-3.5-turbo')
    prediction = result['choices'][0]['message']['content']
    
    return prediction

# Main Function
def generate_dataset(input_file, api_key):
    count = 0
    data = []
    with open(input_file, 'r') as f:
        decoder = json.JSONDecoder()
        text = f.read()
        while text:
            obj, idx = decoder.raw_decode(text)
            data.append(obj)
            text = text[idx:].lstrip()
    
    output_data = []

    for item in data:
        dialog = extract_utterances(item)
        rewritten = rewrite_dialog(api_key, dialog)
        
        output_data.append({
            "source": dialog,
            "rewritten": rewritten
        })
        count += 1
        print('count=',count)

    with open('/cluster/scratch/wangjun/keyword+post_flan-t5-xl_result/combined_gpt3.5_generated_train.json', 'w') as f:
        json.dump(output_data, f, indent=4)

# Example
input_file = '/cluster/scratch/wangjun/keyword+post_flan-t5-xl_result/combined.json'
api_key = ''  # Replace with your API key
generate_dataset(input_file, api_key)
