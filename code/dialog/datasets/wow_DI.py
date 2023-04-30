import copy
import json
import os

import datasets
import numpy as np
from tqdm import tqdm
import random
from .base import DocumentGroundedDataset


class WizardOfWikipedia(DocumentGroundedDataset, datasets.GeneratorBasedBuilder):
    _URL = "http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz"

    def _get_knowledge_documents(self, turn):
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

    def _get_user(self, turn):
        if "wizard" in turn["speaker"].lower():
            return "system"
        else:
            return "user"


    # def _map_to_common_format(self, sample): #original code
    #     context = []
    #     samples = []
    #     # write sample into a txt file called temp.txt

    #     # import ipdb
    #     # ipdb.set_trace()

    #     # print('sample', sample)
    #     with open('/cluster/scratch/feiclu/dialogue_inpainting4_14/temp/sample.txt', 'w') as f:
    #         f.write(str(sample))

    #     for turn in sample["dialog"]:
    #         formatted_turn = {}   

    #         # not annotated in WoW
    #         formatted_turn["dialog_act"] = ""
    #         formatted_turn["text"] = turn["text"]
    #         formatted_turn["user"] = self._get_user(turn)

    #         knowledge_documents = self._get_knowledge_documents(turn)

    #         if len(knowledge_documents) > 0: #and knowledge_documents[0]["text"] != "no_passages_used":
    #             new_sample = {
    #                 "context": copy.deepcopy(context),
    #                 "dataset_id": "WizardOfWikipedia",
    #                 "dialog_act": "",
    #                 "knowledge": knowledge_documents,
    #                 "response": formatted_turn["text"]
    #             }
    #             samples.append(new_sample)

    #         context.append(formatted_turn)
    #     # with open('/cluster/home/wangjun/dialog_inpainting/faithful-dialogue-master/code/dialog/datasets/samples.txt', 'w') as f:
    #     #     f.write(str(samples))
    #     # import ipdb
    #     # ipdb.set_trace()

    #     return samples

    def _map_to_common_format(self,sample):#junling modify
            context = []
            samples = []
            
            random_ratio = 0.3
            
            
            # 0.3 means 30% of the data will be randomly masked
            for turn in sample["dialog"]:
                formatted_turn = {}   

                # not annotated in WoW
                formatted_turn["dialog_act"] = ""
                formatted_turn["text"] = "<extra_id_0>" if random.random() < random_ratio else turn["text"]
                formatted_turn["user"] = self._get_user(turn)

                knowledge_documents = self._get_knowledge_documents(turn)

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
            #create a directory to store the samples
            
          
            with open('/cluster/scratch/feiclu/dialogue_inpainting4_14/samples.txt', 'w') as f:
                f.write(str(samples))

            return samples

    def _split_generators(self, dl_manager):
        url_to_download = self._URL
        data_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
        splits = ["train", "val", "test"]
        file_names = ["train.json", "valid_random_split.json", "test_random_split.json"]
        data_files = {
            split: os.path.join(data_path, file_name) for split, file_name in zip(splits, file_names)
        }

        formatted_data = {}

        for split in splits:
            with open(data_files[split], "r") as f:
                data = json.load(f)

            formatted_data[split] = [self._map_to_common_format(sample) for sample in tqdm(data)]

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": formatted_data[split]
                })
            for ds_split, split in
            zip([datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST], splits)
        ]
