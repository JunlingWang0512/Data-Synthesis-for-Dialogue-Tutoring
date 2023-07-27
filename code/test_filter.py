import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from random import randrange
from utils import NumpyEncoder
import json
import itertools
from datasets import Dataset, load_dataset, concatenate_datasets
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

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

# This dictionary will hold all the TF-IDF scores
tfidf_scores = defaultdict(int)
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize')
with open('/cluster/scratch/wangjun/local_data/book_dataset_v4/math/algebra_and_trigonometry.json') as f:
    dataset = json.load(f)

# Prepare data for TF-IDF vectorizer
text = []
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




nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize')
nlp_spacy = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
STOPWORDS_THRESHOLD = 0.5  # sentences with more stop words than this threshold will be removed

# def is_informative(sentence):
#     doc = nlp_spacy(sentence)
#     if not doc.ents:
#         return False  # if no named entities, not informative
#     num_stopwords = sum([token.is_stop for token in doc])
#     if num_stopwords / len(doc) > STOPWORDS_THRESHOLD:
#         return False  # if too many stop words, not informative
#     return True
# def is_informative(sentence):
#     doc = nlp_spacy(sentence)
#     num_stopwords = sum([token.is_stop for token in doc])
#     if num_stopwords / len(doc) > STOPWORDS_THRESHOLD:
#         return False  # if too many stop words, not informative
#     return True

# def is_informative(sentence):
#     doc = nlp_spacy(sentence)
#     # Exclude sentences that don't contain any domain-specific terms
#     # if not any(term in sentence for term in DOMAIN_SPECIFIC_TERMS):
#     #     return False
#     # Exclude sentences that are questions
#     if sentence.endswith("?"):
#         return False
#     num_stopwords = sum([token.is_stop for token in doc])
#     # Exclude sentences that contain too many common, non-domain-specific words
#     if num_stopwords / len(doc) > STOPWORDS_THRESHOLD:
#         return False
#     return True

def is_informative(sentence, THRESHOLD):
    doc = nlp_spacy(sentence)
    if not doc.ents and all(tfidf_scores.get(token.lemma_.lower(), 0) < THRESHOLD for token in doc):
        print('tf_idf')
        return False
    num_stopwords = sum([token.is_stop for token in doc])
    if num_stopwords / len(doc) > STOPWORDS_THRESHOLD:
        print('stop_word',num_stopwords / len(doc))
        return False
    return True

# s = 'An airline pilot maneuvers a plane toward a narrow runway'
print('newsentence')
s = 'These functions are the reciprocals of the first three functions.'
print(is_informative(s,7))
doc = nlp_spacy(s)
# for token in doc:
#     print(token,tfidf_scores.get(token.lemma_.lower(), 0))
# print(is_informative(s, 0.1))

print('newsentence')
s = 'These functions are the reciprocals of the first three functions.'
doc = nlp_spacy(s)
for token in doc:
    print(token,tfidf_scores.get(token.lemma_.lower(), 0))

# print('newsentence')
# s = 'We can graph $$\\(y = \\cot\\; x\\)$$ by observing the graph of the tangent function because these two functions are reciprocals of one another.'
# doc = nlp_spacy(s)
# for token in doc:
#     print(token,tfidf_scores.get(token.lemma_.lower(), 0))

# print('newsentence')
# s = 'Where the graph of the tangent function decreases, the graph of the cotangent function increases.'
# doc = nlp_spacy(s)
# for token in doc:
#     print(token,tfidf_scores.get(token.lemma_.lower(), 0))

# 5-9
# 用7