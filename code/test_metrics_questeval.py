import questeval
from questeval.questeval_metric import QuestEval
questeval = QuestEval(no_cuda=True)


import json
import pandas as pd
# import openai
import re
import csv
import random
# import nltk
# from nltk.util import ngrams
# from nltk.probability import FreqDist
# from scipy.stats import entropy
from collections import Counter
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import Counter
# import requests
import time
# from transformers import pipeline
# import difflib
# # from transformers import pipeline, AutoModelForSeq2SeqLM,AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForSequenceClassification
# import difflib
# import nltk
# from ast import literal_eval
# from transformers import pipeline, AutoTokenizer, 
# import torch
import pandas as pd





file_path = '/cluster/scratch/wangjun/keyword+post_flan-t5-xl_result/math_search_output_post.json'

data = []
with open(file_path, 'r') as f:
    decoder = json.JSONDecoder()
    text = f.read()
    while text:
        obj, idx = decoder.raw_decode(text)
        data.append(obj)
        text = text[idx:].lstrip()

# Initialize a list to store the dialog texts and their corresponding scores
score_data = []

# randomly select 20 dialogs for evaluation
# random.seed(123)
# selected_data = random.sample(data, 100)
selected_data = data


import os
OUTPUT_FILE = '/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/output/8_24_questeval/flant5xl_math_questeval.csv'
SAVE_INTERVAL = 10

# Initialize an output file
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["dialog_index", "tuple", "QFactScore"])

def get_starting_point():
    if os.path.exists(OUTPUT_FILE):
        try:
            # Load the CSV and determine the last dialog index processed
            existing_df = pd.read_csv(OUTPUT_FILE)
            if 'dialog_index' in existing_df.columns and not existing_df['dialog_index'].isna().all():
                last_processed_index = existing_df['dialog_index'].max()
                
                # Explicitly handle the case where the max() returns NaN
                if pd.isna(last_processed_index):
                    return 0
                
                return int(last_processed_index) + 1
        except pd.errors.EmptyDataError:
            # If the file is empty, start from the beginning
            return 0
    return 0


start_index = get_starting_point()
print('start index = ',start_index)
processed_dialogs_count = 0

for index, obj in enumerate(selected_data[start_index:], start=start_index):
    dialog = obj['utterances']
    sentences = obj['sentences']

    if dialog == []:continue
    tuples = [dialog[i:i+2] for i in range(0, len(dialog), 2)]
    for tuple_ in tuples:
        score_dict = {}
        score_dict['dialog_index'] = index
        score_dict['tuple'] = tuple_
        # print('ok')
        score = questeval.corpus_questeval(hypothesis=[tuple_[0]], sources=[tuple_[1]])
        score_dict['QuestEval'] = score['corpus_score']
        score_data.append(score_dict)

    processed_dialogs_count += 1
    print('processed_dialogs_count = ',processed_dialogs_count)

    # Append results every SAVE_INTERVAL dialogs processed
    if processed_dialogs_count % SAVE_INTERVAL == 0:
        temp_df = pd.DataFrame(score_data)
        temp_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        print('processed_dialogs_count = ',processed_dialogs_count)
        # Clear the score_data to avoid duplicating entries
        score_data.clear()

# After processing all data, if there's still remaining data in score_data, append it to the file
if score_data:
    temp_df = pd.DataFrame(score_data)
    temp_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

# Compute the average scores and print the results
df = pd.read_csv(OUTPUT_FILE)
avg_scores = df[['QuestEval']].mean(skipna=True)
df.loc['Average'] = [None, 'Average'] + avg_scores.tolist()
print(df)
df.to_csv(OUTPUT_FILE, index=False)