from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
from scipy.spatial.distance import cosine
import textstat
import json
import pandas as pd
import openai
import re
import csv
import random
import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
from scipy.stats import entropy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import requests
import time
from transformers import pipeline
import difflib
from transformers import pipeline, AutoModelForSeq2SeqLM,AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForSequenceClassification
import difflib
import nltk
from ast import literal_eval
# from transformers import pipeline, AutoTokenizer, 
import torch
import pandas as pd
import os
import spacy

qa_model = pipeline('question-answering', model = 'deepset/deberta-v3-large-squad2')

nli_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli')
nli_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')


def sentences_match(sentence1: str, sentence2: str) -> float:
    # Tokenize both sentences
    tokens1 = nltk.word_tokenize(sentence1)
    tokens2 = nltk.word_tokenize(sentence2)

    # Initialize a SequenceMatcher with the tokens
    sequence_matcher = difflib.SequenceMatcher(None, tokens1, tokens2)

    # Get the similarity ratio
    similarity_score = sequence_matcher.ratio()

    return similarity_score

def Q_A_eval5_entailment_score(question: str, given_answer: str, passage: str):
    # Initialize the QA model
    repetition = sentences_match(question,given_answer)
    if not question:
        print('Empty question was given as input.')
        return 0

    # Join all sentences into a single context
    context = passage

    # Use the QA model to predict the answer from the context
    output = qa_model(question=question, context=context)
    print('qa_output',output)
    # Store the predicted answer
    predicted_answer = output['answer']

    if not predicted_answer.strip():  #if cannot predict any answer, return 0 #or output['score'] < some_threshold:
        return 0
    # Check the token-level similarity
    similarity_score = sentences_match(given_answer, predicted_answer)
    if similarity_score > 0.8:
        return 1

    # Prepare the inputs for the NLI model
    premise = given_answer
    hypothesis = predicted_answer

    # Encode the inputs
    inputs = nli_tokenizer.encode_plus(premise, hypothesis, return_tensors='pt')

    # Get the model's predictions
    outputs = nli_model(**inputs)[0]

    # Get the probabilities by applying the softmax function
    probs = torch.nn.functional.softmax(outputs, dim=-1)

    # Compute probability of entailment and non-entailment
    entailment_prob = probs[0, 2].item()
    non_entailment_prob = probs[0, 0].item() + probs[0, 1].item()

    return entailment_prob


def extract_score(text):
    match = re.search('Score:\s*(\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1))
    else:
        return None

# Initialize models and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Initialize language tool
# tool = language_tool_python.LanguageTool('en-US')

def get_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.pooler_output[0].detach().numpy()


def calculate_similarity(text1, text2):
    vec1 = get_vector(text1)
    vec2 = get_vector(text2)
    return 1 - cosine(vec1, vec2)


# Assuming the first statement was the AI question and the second the text, check relevance
def flesch_reading_ease(dialogue): #fluent score
    # Concatenating all sentences in the dialogue into a single text
    dialogue_text = " ".join(dialogue)
    # Calculate the Flesch Reading Ease score
    flesch_reading_ease = textstat.flesch_reading_ease(dialogue_text)
    # Scale to 0-10, where 10 is the easiest text to read
    # scaled_score = flesch_reading_ease / 10
    # # Clamp the score to the maximum of 10
    # scaled_score = min(10, scaled_score)
    return flesch_reading_ease

def relevance_score(dialogue):
    local_relevance_scores = []
    for i in range(len(dialogue)-1):
        local_relevance_scores.append(calculate_similarity(dialogue[i], dialogue[i+1]))
    scaled_score = (sum(local_relevance_scores)/len(local_relevance_scores)) * 10
    return scaled_score

file_path = '/cluster/scratch/wangjun/dialog_rewritting/result/model2_rewrite_flant5xl_math_8_20.json'

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
count = 0
# for obj in selected_data:
#     dialog = obj['utterances']
#     sentences = obj['sentences'] #used by q&a prediction model
#     count += 1
#     print('dialog count = ', count)
#     # print(dialog)
#     if dialog == []:continue
#     tuples = [dialog[i:i+2] for i in range(0, len(dialog), 2)]
#     for tuple_ in tuples:
#         score_dict = {}
#         score_dict['tuple'] = tuple_
#         # score_dict['relevance'] = relevance_score(tuple_)
#         # score_dict['flesch_reading_ease'] = flesch_reading_ease(tuple_)
#         # score_dict['bigram_entropy'] = calculate_bigram_entropy(tuple_)
#         # score_dict['is_general'] = is_general_question(tuple_[0])
#         # score_dict['distinct_3'] = distinct_3(tuple_[0])
#         # score_dict['toxicity'] = calculate_toxicity(tuple_[0])
#         score_dict['QFactScore'] = Q_A_eval5_entailment_score(tuple_[0],tuple_[1],sentences)
#         # score_dict['correctness'] = calculate_correctness_score(tuple_)
#         # score_dict['coherence'] = calculate_coherence_score(tuple_)
#         # Append the dictionary to the score_data list
#         score_data.append(score_dict)

# # Convert the list of dictionaries to a DataFrame
# df = pd.DataFrame(score_data)

# df.to_csv('gpt3.5_social_metrics.csv')
# # Calculate the average scores and append them to the DataFrame
# # df['toxicity'] = df['toxicity'].replace({None: pd.NA}).astype(float)  # replace None with NaN and ensure data type is float
# # avg_scores = df[['relevance', 'flesch_reading_ease', 'bigram_entropy', 'is_general', 'distinct_3', 'toxicity','Q_A_eval']].mean(skipna=True)
# avg_scores = df[['QFactScore']].mean(skipna=True)
# df.loc['Average'] = ['Average'] + avg_scores.tolist()

# print(df)
# df.to_csv('gpt3.5_social_metrics.csv')


# Constants
OUTPUT_FILE = '/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/output/8_16_qfactscore_calculation/rewrite2_flant5xl_math_metrics_8_20.csv'
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
    # dialog = obj['utterances']
    dialog = obj['rewritten']
    # sentences = obj['sentences']
    passage = obj['passage']

    if dialog == []:continue
    
    
    # Adjust here to ensure only pairs of turns are considered
    if len(dialog) % 2 == 1:
        dialog = dialog[:-1]  # Remove the last turn if the number of turns is odd

    tuples = [dialog[i:i+2] for i in range(0, len(dialog), 2)]
    # tuples = [dialog[i:i+2] for i in range(0, len(dialog), 2)]
    
    for tuple_ in tuples:
        score_dict = {}
        score_dict['dialog_index'] = index
        score_dict['tuple'] = tuple_
        score_dict['QFactScore'] = Q_A_eval5_entailment_score(tuple_[0],tuple_[1],passage)
        score_data.append(score_dict)

    processed_dialogs_count += 1

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
avg_scores = df[['QFactScore']].mean(skipna=True)
df.loc['Average'] = [None, 'Average'] + avg_scores.tolist()
print(df)
df.to_csv(OUTPUT_FILE, index=False)
#####################################################################################

# file_path = '/cluster/scratch/wangjun/dialog_rewritting/result/rewrite_flant5xl_business.json'

# data = []
# with open(file_path, 'r') as f:
#     decoder = json.JSONDecoder()
#     text = f.read()
#     while text:
#         obj, idx = decoder.raw_decode(text)
#         data.append(obj)
#         text = text[idx:].lstrip()

# # Initialize a list to store the dialog texts and their corresponding scores
# score_data = []


# selected_data = data
# count = 0

# # Constants
# OUTPUT_FILE = '/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/output/8_16_qfactscore_calculation/source2_flant5xl_business_metrics_8_19.csv'
# SAVE_INTERVAL = 10

# # Initialize an output file
# if not os.path.exists(OUTPUT_FILE):
#     with open(OUTPUT_FILE, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["dialog_index", "tuple", "QFactScore"])

# def get_starting_point():
#     if os.path.exists(OUTPUT_FILE):
#         try:
#             # Load the CSV and determine the last dialog index processed
#             existing_df = pd.read_csv(OUTPUT_FILE)
#             if 'dialog_index' in existing_df.columns and not existing_df['dialog_index'].isna().all():
#                 last_processed_index = existing_df['dialog_index'].max()
                
#                 # Explicitly handle the case where the max() returns NaN
#                 if pd.isna(last_processed_index):
#                     return 0
                
#                 return int(last_processed_index) + 1
#         except pd.errors.EmptyDataError:
#             # If the file is empty, start from the beginning
#             return 0
#     return 0


# start_index = get_starting_point()
# print('start index = ',start_index)
# processed_dialogs_count = 0

# for index, obj in enumerate(selected_data[start_index:], start=start_index):
#     # dialog = obj['utterances']
#     dialog = obj['source']
#     # sentences = obj['sentences']
#     passage = obj['passage']

#     if dialog == []:continue
    
    
#     # Adjust here to ensure only pairs of turns are considered
#     if len(dialog) % 2 == 1:
#         dialog = dialog[:-1]  # Remove the last turn if the number of turns is odd

#     tuples = [dialog[i:i+2] for i in range(0, len(dialog), 2)]
#     # tuples = [dialog[i:i+2] for i in range(0, len(dialog), 2)]
    
#     for tuple_ in tuples:
#         score_dict = {}
#         score_dict['dialog_index'] = index
#         score_dict['tuple'] = tuple_
#         score_dict['QFactScore'] = Q_A_eval5_entailment_score(tuple_[0],tuple_[1],passage)
#         score_data.append(score_dict)

#     processed_dialogs_count += 1

#     # Append results every SAVE_INTERVAL dialogs processed
#     if processed_dialogs_count % SAVE_INTERVAL == 0:
#         temp_df = pd.DataFrame(score_data)
#         temp_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
#         print('processed_dialogs_count = ',processed_dialogs_count)
#         # Clear the score_data to avoid duplicating entries
#         score_data.clear()

# # After processing all data, if there's still remaining data in score_data, append it to the file
# if score_data:
#     temp_df = pd.DataFrame(score_data)
#     temp_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

# # Compute the average scores and print the results
# df = pd.read_csv(OUTPUT_FILE)
# avg_scores = df[['QFactScore']].mean(skipna=True)
# df.loc['Average'] = [None, 'Average'] + avg_scores.tolist()
# print(df)
# df.to_csv(OUTPUT_FILE, index=False)