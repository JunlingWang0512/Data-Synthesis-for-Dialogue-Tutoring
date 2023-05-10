import copy
import json
import os

import datasets
import numpy as np
from tqdm import tqdm

# from .base import DocumentGroundedDataset


class QRECC():
    # _URL = "http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz"
    _URL = '/cluster/scratch/wangjun/local_data/QRECC'


    
    def _get_knowledge_documents(self, turn):
        docs = []
        return docs
    
    def _get_user(self, sample):
        return None
    
# one turn have both question and answer
    def _map_to_common_format(self, dialogue):
        context = []
        samples = []

        for turn in dialogue:
            formatted_turn = {}

            if 'Question' in turn:
                formatted_turn["dialog_act"] = ""
                formatted_turn["text"] = turn["Question"]
                formatted_turn["user"] = "user"
                context.append(formatted_turn)

            if 'Answer' in turn:
                formatted_turn = {}
                formatted_turn["dialog_act"] = ""
                formatted_turn["text"] = turn["Answer"]
                formatted_turn["user"] = "system"

                knowledge_documents = self._get_knowledge_documents(turn)

                # if len(knowledge_documents) > 0:
                new_sample = {
                    "context": copy.deepcopy(context),
                    "dataset_id": "QRECC",
                    "dialog_act": "",
                    "knowledge": knowledge_documents,
                    "response": formatted_turn["text"]
                }
                samples.append(new_sample)

                context.append(formatted_turn)

        return samples

    def _split_generators(self):  # junling modify done 
        # url_to_data = self._URL
        # data_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
        #dl_manager is used for download, not used here.
        data_path = self._URL
        splits = ["train", "val", "test"]
        file_names = ["qrecc_train2.json", "validation.json", "qrecc_test.json"]
        data_files = {   
            split: os.path.join(data_path, file_name) for split, file_name in zip(splits, file_names)
        }

        formatted_data = {}

        for split in splits:
            with open(data_files[split], "r") as f:
                data = json.load(f)  #到这里
                # formatted_data[split] = [self._map_to_common_format(sample) for sample in tqdm(data)] #ORIGINAL
                # Identify when a new dialogue begins based on the last three digits in the "qid" field #junling modify
                dialogue = [] #debug
                formatted_data[split] = []
                for sample in tqdm(data):
                    # print('sample = ',sample)
                    if sample["Turn_no"] == 1 and len(dialogue) > 0:
                        # print('dialogue = ',dialogue)
                        formatted_samples = self._map_to_common_format(dialogue)
                        formatted_data[split].append(formatted_samples)
                        dialogue = []

                    dialogue.append(sample)

                # Process the last dialogue
                if len(dialogue) > 0:
                    formatted_samples = self._map_to_common_format(dialogue)
                    formatted_data[split].append(formatted_samples)

        return formatted_data['test']



instance = QRECC()
result = instance._split_generators()
with open('/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/dialog/datasets/formatted_data[test]_qrecc.txt','w') as f:
    f.write(str(result))
