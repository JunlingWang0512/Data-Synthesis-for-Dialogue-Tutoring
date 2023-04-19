import copy
import random
def _get_user(turn):#junling modify
        if "wizard" in turn["speaker"].lower():
            return "system"
        else:
            return "user"
def _get_knowledge_documents(turn):#junling modify
    docs = []

    if "checked_sentence" in turn:
        if len(turn["checked_sentence"].values()) == 0:
            text = ""
        else:
            text = list(turn["checked_sentence"].values())[0]
        doc = {
            "description": "checked_sentence",
            "text": text
        }
        docs.append(doc)

    return docs

def _map_to_common_format7(sample):
        context = []
        samples = []
        
        random_ratio = 0.8 # 0.3 means 30% of the data will be randomly masked
        for turn in sample["dialog"]:
            formatted_turn = {}   

            # not annotated in WoW
            formatted_turn["dialog_act"] = ""
            formatted_turn["text"] = "<extra_id_0>" if random.random() < random_ratio else turn["text"]
            formatted_turn["user"] = _get_user(turn)

            knowledge_documents = _get_knowledge_documents(turn)

            if len(knowledge_documents) > 0: #and knowledge_documents[0]["text"] != "no_passages_used":
                new_sample = {
                    "context": copy.deepcopy(context),
                    "dataset_id": "WizardOfWikipedia",
                    "dialog_act": "",
                    "knowledge": knowledge_documents,
                    "response": formatted_turn["text"]
                }
                samples.append(new_sample)

            context.append(formatted_turn)
        # with open('/cluster/home/wangjun/dialog_inpainting/faithful-dialogue-master/code/dialog/datasets/samples.txt', 'w') as f:
        #     f.write(str(samples))
        # import ipdb
        # ipdb.set_trace()

        return samples



with open('sample.txt', 'r') as f:
    content = f.read()
    sample = eval(content)



samples = _map_to_common_format7(sample)
# print('samples', samples)
# save the samples into a txt file
with open('samples_7.txt', 'w') as f:
    f.write(str(samples))