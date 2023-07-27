import json

def read_dialogues(file_path):
    decoder = json.JSONDecoder()
    data = []
    with open(file_path, 'r') as f:
        text = f.read()
        while text:
            obj, idx = decoder.raw_decode(text)
            data.append(obj)
            text = text[idx:].lstrip()

    for obj in data:
        dialog = obj['utterances']
        dialog_format = []
        for i in range(0, len(dialog), 2):
            dialog_format.append('Teacher asks: '+dialog[i])
            dialog_format.append('Student answers: '+dialog[i+1])
        print('________new dialog___________________________________________-')
        print(dialog_format)

file_path = '/cluster/scratch/wangjun/dialogue_inpainting5_18_flan_lora_xl/work/ukp/huggingface/6_27_algebra_search/HuggingfaceSearchJob.8FtuwoC7t0JX/output/search_output_post.json'
read_dialogues(file_path)
