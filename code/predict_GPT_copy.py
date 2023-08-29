import openai
import re
import csv

openai.api_key = "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"

from typing import List

def prepare_prompt(sentences: List[str], document_title: str) -> str:
    if len(sentences) % 2 == 0:
        raise ValueError("The input 'sentences' must have an odd number of elements.")
    

    preprompt = f"Assuming you are a teacher asking questions to student to help student learn a passage of textbook about {document_title}. Given the following dialog, try to ask a question to fulfill the <mask> place, please consider the context and only provide me with the question text:\n\n"
    
    for i in range(len(sentences) - 1):  # Exclude the last sentence
        if i % 2 == 0:  # Teacher's turn to speak
            preprompt += f"Teacher: {sentences[i]}\n"
        else:  # Student's turn to speak
            preprompt += f"Student: {sentences[i]}\n"

    # For the last Teacher's turn, put a "<mask>"
    preprompt += "Teacher: <mask>\n"
    # Add the last student's answer
    preprompt += f"Student: {sentences[-1]}\n"

    return preprompt
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

# from dialog.main import RunMode, main

# if __name__ == "__main__":
#     main(RunMode.PREDICT)
import torch
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from random import randrange
from utils import NumpyEncoder
import json
import itertools
# from datasets import Dataset, load_dataset, concatenate_datasets
import time
import random
from typing import List, Tuple
# import nltk
from uuid import uuid4
import sys
import stanza
import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords





#tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize')
nlp_spacy = spacy.load("en_core_web_sm")

# Define custom tokenizer for using lemma
def custom_tokenizer(document):
    doc = nlp_spacy(document)
    return [token.lemma_.lower() for token in doc]
# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, stop_words='english', lowercase=True)

# This dictionary will hold all the TF-IDF scores
tfidf_scores = defaultdict(int)
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize')
# /cluster/scratch/wangjun/local_data/book_dataset_v4/business/business_ethics.json
# /cluster/scratch/wangjun/local_data/book_dataset_v4/math/algebra_and_trigonometry.json
with open('/cluster/scratch/wangjun/local_data/book_dataset_v4/social_sciences/psychology_2e.json') as f:
    dataset = json.load(f)

# Prepare data for TF-IDF vectorizer
text = []
tfidf_scores = defaultdict(float)
for key in dataset:
    if key not in ('book_statistics', 'chapter_concepts', 'chapter_questions'):
        section = dataset[key]['content']
        for paragraph in section:
            doc = nlp_stanza(paragraph)
            sentences = [sentence.text for sentence in doc.sentences]
            text.extend(sentences)

# Calculate TF-IDF scores
tfidf_matrix = tfidf_vectorizer.fit_transform(text)
feature_names = tfidf_vectorizer.get_feature_names_out()

for sentence_vector in tfidf_matrix:
    for term_id, score in zip(sentence_vector.indices, sentence_vector.data):
        tfidf_scores[feature_names[term_id]] += score
#tf-idf



stop_words = set(stopwords.words('english'))

# MIN_LEN = 5   # minimum sentence length
# MAX_LEN = 50  # maximum sentence length
STOPWORDS_THRESHOLD = 0.7  # sentences with more stop words than this threshold will be removed

def is_informative(sentence, THRESHOLD):
    if sentence.endswith("?"):
        return False
    doc = nlp_spacy(sentence)
    if not doc.ents and all(tfidf_scores.get(token.lemma_.lower(), 0) < THRESHOLD for token in doc):
        return False
    num_stopwords = sum([token.is_stop for token in doc])
    if num_stopwords / len(doc) > STOPWORDS_THRESHOLD:
        return False
    return True

count_section = 0
for key in dataset:
    
    if key not in ('book_statistics','chapter_concepts','chapter_questions'):
        count_section += 1
        section = dataset[key]['content']
        count = 0
        result = []
        document_title = str(key)
        for paragraph in section:
            # sentences = nltk.sent_tokenize(paragraph)
            # replace nltk sentence tokenization with deepsegment
            # sentences = segmenter.segment(paragraph)
            doc = nlp_stanza(paragraph)
            sentences = [sentence.text for sentence in doc.sentences]

            original_sentences = sentences.copy()
            if(len(sentences) == 1):
                document_title = str(sentences[0])
                
            elif(len(sentences) > 1):
                # Filter sentences before generating dialogs
                # MIN_LEN <= len(s.split()) <= MAX_LEN and
                sentences = [s for s in sentences if is_informative(s,7)]
                dialog = []  # Initialize an empty dialog
                author_num = []
                # test_datasets = []

                # Generate dialog inpainting
                for idx, sentence in enumerate(sentences):
                    if idx == 0:
                        prompt = prepare_prompt([sentence], document_title)
                    else:
                        prompt = prepare_prompt(dialog + [sentence], document_title)
                    
                   #需要修改
                    # print('test_dataset:',test_dataset)
                    # gpt-3.5-turbo
                    # GPT-4-0314
                    # GPT-4-0613
                    result = generate_response0(prompt,'gpt-3.5-turbo')
                    prediction = result['choices'][0]['message']['content']
                    
                    # process the prediction results
                    prediction = prediction.replace('<user>', '')
                    prediction = prediction.replace('user>', '')
                    prediction = prediction.replace('<user', '')
                    prediction = prediction.replace('system>', '')
                    prediction = prediction.strip()

                    
                    generated_sentence = prediction

                    dialog.append(generated_sentence)  # Add the generated sentence as the first element
                    dialog.append(sentence)  # Add the current input sentence
                    author_num.append(0)
                    author_num.append(1)
                            
                # Save the targeted content
                output_data = {
                    "title": document_title,
                    "pid": str(uuid4()),
                    "passage": " ".join(original_sentences),
                    "sentences": original_sentences,
                    "author_num": author_num,
                    "utterances": dialog
                }
                print(output_data)
                prediction_output_file = '/cluster/scratch/wangjun/GPT_results/GPT_3.5/social_search_output.json'
                with open(prediction_output_file, 'a') as f:
                        json.dump(output_data, f, cls=NumpyEncoder)
