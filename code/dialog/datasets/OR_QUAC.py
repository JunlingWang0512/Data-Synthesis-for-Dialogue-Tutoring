import copy
import json
import os

import datasets
import numpy as np
from tqdm import tqdm

from .base import DocumentGroundedDataset


class OR_QUAC(DocumentGroundedDataset, datasets.GeneratorBasedBuilder):
    # _URL = "http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz"
    _URL = '/cluster/scratch/wangjun/local_data/OR_QUAC'


    
    def _get_knowledge_documents(self, turn):
        docs = []
        if "evidences" in turn:
            if len(turn['evidences']) == 0:
                text = ""
            else:
                text = list(turn["evidences"])
            doc = {
                "description": "evidences",
                "text":text
            }
            docs.append(doc)

        return docs
    
    def _get_user(self, sample):
        return None
    
# one turn have both question and answer
    def _map_to_common_format(self, dialogue):
        context = []
        samples = []

        for turn in dialogue:
            formatted_turn = {}

            if 'question' in turn:
                formatted_turn["dialog_act"] = ""
                formatted_turn["text"] = turn["question"]
                formatted_turn["user"] = "user"
                context.append(formatted_turn)

            if 'answer' in turn:
                formatted_turn = {}
                formatted_turn["dialog_act"] = ""
                formatted_turn["text"] = turn["answer"]['text']
                formatted_turn["user"] = "system"

                knowledge_documents = self._get_knowledge_documents(turn)

                if len(knowledge_documents) > 0:
                    new_sample = {
                        "context": copy.deepcopy(context),
                        "dataset_id": "OR_QUAC",
                        "dialog_act": "",
                        "knowledge": knowledge_documents,
                        "response": formatted_turn["text"]
                    }
                    samples.append(new_sample)

                context.append(formatted_turn)

        return samples

    def _split_generators(self, dl_manager):  # junling modify done 
        # url_to_data = self._URL
        # data_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
        #dl_manager is used for download, not used here.
        data_path = self._URL
        splits = ["train", "val", "test"]
        file_names = ["train_filtered.json", "validation_filtered.json", "test_filtered.json"]
        data_files = {   
            split: os.path.join(data_path, file_name) for split, file_name in zip(splits, file_names)
        }

        formatted_data = {}

        for split in splits:
            with open(data_files[split], "r") as f:
                data = json.load(f)
                # formatted_data[split] = [self._map_to_common_format(sample) for sample in tqdm(data)] #ORIGINAL
                # Identify when a new dialogue begins based on the last three digits in the "qid" field #junling modify
                dialogue = [] #debug
                formatted_data[split] = []
                for sample in tqdm(data):
                    # print('sample = ',sample)
                    if sample["qid"][-3:] == "q#0" and len(dialogue) > 0:
                        # print('dialogue = ',dialogue)
                        formatted_samples = self._map_to_common_format(dialogue)
                        formatted_data[split].append(formatted_samples)
                        dialogue = []

                    dialogue.append(sample)

                # Process the last dialogue
                if len(dialogue) > 0:
                    formatted_samples = self._map_to_common_format(dialogue)
                    formatted_data[split].append(formatted_samples)

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": formatted_data[split]
                })
            for ds_split, split in
            zip([datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST], splits)
        ]




