import random
# 手动改这个程序吧
def mask_utterance(data, mark_ratio=0.3):
    masked_data = []

    for i in range(len(data['id'])):
        marked = False
        id = data['id'][i]
        context = data['context'][i]
        response = data['response'][i]

        if random.random() < mark_ratio:
            marked = True
            last_text = context[-1]['text']
            context[-1]['text'] = "<extra_id_0>"
            response = "s <extra_id_0>"

        masked_data.append({
            'id': id,
            'context': context,
            'response': response,
            'marked': marked
        })

    for i in range(len(masked_data) - 1):
        if masked_data[i]['marked']:
            prev_response = masked_data[i]['response']
            next_context = masked_data[i + 1]['context']

            for c in next_context:
                if c['text'] == prev_response:
                    c['text'] = "s <extra_id_0>"

    return masked_data


with open('features_preprocess_input.txt', 'r') as f:
        content = f.read()
        features = eval(content)

# dialogue_data = list(zip(
#     features["context"],
#     features["dialog_act"],
#     features["knowledge"],
#     features["response"]
# ))

dialogue_data = [
    {
        "id": i,
        "context": context,
        "dialog_act": dialog_act,
        "knowledge": knowledge,
        "response": response
    }
    for i, (context, dialog_act, knowledge, response) in enumerate(zip(
        features["context"],
        features["dialog_act"],
        features["knowledge"],
        features["response"]
    ))
]



masked_data = mask_utterance(dialogue_data)
