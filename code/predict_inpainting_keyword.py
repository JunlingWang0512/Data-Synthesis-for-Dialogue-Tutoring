#predict_inpainting_keyword

# from dialog.main import RunMode, main

# if __name__ == "__main__":
#     main(RunMode.PREDICT)
import re
from rake_nltk import Rake
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

# from deepsegment import DeepSegment
# nltk.download('punkt')

# Load peft config for pre-trained checkpoint etc.
config_path = sys.argv[1]

with open(config_path, 'r') as f:
    config = json.load(f)

# Access model_path
peft_model_id = config['model_name_or_path']
prediction_output_file = config['prediction_output_file']
# peft_model_id = "/cluster/scratch/wangjun/dialogue_inpainting5_18_flan_lora_xl/work/ukp/huggingface/training/HuggingfaceTrainingJob.wrncuVcHOHOI/output/models/epoch-best"
config = PeftConfig.from_pretrained(peft_model_id)

# load base LLM model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,  device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map='auto')
#  load_in_8bit=True, device_map="auto" #junling modify
model.eval()

# print("Peft model loaded")

# tokenizer = AutoTokenizer.from_pretrained(  #can be merged into main function if needed later
#         'google/flan-t5-base',
#         cache_dir='/cluster/scratch/wangjun/dialogue_inpainting5_18_flan_lora_xl/cache',
#         use_fast=True,
#         revision="main",
#         use_auth_token=None,
#     )
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
def _truncate_to_max_length(context):
        context_len = 0
        truncated_context = []
        history_max_utterances = 999
        history_max_tokens = 384

        context = context[-history_max_utterances:]
        for turn in context[::-1]:
            if context_len + len(turn) < history_max_tokens:
                truncated_context.append(turn)
                context_len += len(turn)
            else:
                break

        return truncated_context[::-1]
def _tokenize_with_special_tokens(turn):
        user_token = "<user>"
        system_token ="<system>"
        dialog_act = tokenizer(turn["dialog_act"], add_special_tokens=False)["input_ids"]
        user_tag = user_token if turn["user"] == "user" else system_token
        user_tag = tokenizer(user_tag, add_special_tokens=False)["input_ids"]
        text = tokenizer(turn["text"], add_special_tokens=False)["input_ids"]
        return user_tag + dialog_act + text
def _process_dialog_context(context):
        context = [_tokenize_with_special_tokens(turn) for turn in context]
        context = _truncate_to_max_length(context)
        return list(itertools.chain.from_iterable(context))
def _process_response(response):
        response = tokenizer(response)["input_ids"]
        if len(response) > 512:
            response = response[:511] + [response[-1]]
        return response
def postprocess_predictions(p, dataset):
        # model_class = self.get_model_class(self.config)

        p.predictions[p.predictions == -100] = tokenizer.pad_token_id
        out = tokenizer.batch_decode(
            p.predictions, skip_special_tokens=True
        )
        return out

def generate_partial_dialog(sentences: List[str], document_title: str, keywords: List[str]) -> Tuple[List[dict], str]:
    sequences, labels = [], []
    
    if len(sentences) % 2 == 0:
        raise ValueError("The input 'sentences' must have an odd number of elements.")

    # Create a string of the keywords for the introduction
    keywords_str = ', '.join(keywords[:-1]) + ' and ' + keywords[-1] if len(keywords) > 1 else keywords[0]

    introduction = f"As your teacher, I'll be asking you very specific questions about the content in this document, particularly related to {keywords_str}. Now, let's start: what's the title of our study material?"
    title_answer = f"The title of our study material is {document_title}"
    dialog = [{'dialog_act': '', 'text': introduction, 'user': 'system'}, {'dialog_act': '', 'text': title_answer, 'user': 'user'}]


    # introduction = f"As your teacher, I'll be asking you very specific questions about the content in this document. For example, if the content is about the Pythagorean theorem, I might ask: 'What is the relationship among the three sides of a right-angled triangle according to the Pythagorean theorem?' Now, let's start: what's the title of our study material?"
    # title_answer = f"The title of our study material is {document_title}"
    # dialog = [{'dialog_act': '', 'text': introduction, 'user': 'system'}, {'dialog_act': '', 'text': title_answer, 'user': 'user'}]


    for i, text in enumerate(sentences[:-1]):
        user = 'system' if i % 2 == 0 else 'user'
        dialog.append({'dialog_act': '', 'text': text, 'user': user})

    dialog.append({'dialog_act': '', 'text': '<extra_id_0>', 'user': 'system'})
    dialog.append({'dialog_act': '', 'text': sentences[-1], 'user': 'user'})

    dialog_act = ''
    
    
    unique_id = f"{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    new_dict = {
    'id': [unique_id],
    'context': [dialog],
    'dataset_id':['dialog_inpainting'],
    'dialog_act':[''],
    'knowledge':[[]],
    'response': ['']
    # 'keywords': [', '.join(keywords)] 
    }
    
    context = _process_dialog_context(dialog)
    
    dialog_act = tokenizer(dialog_act, add_special_tokens=False)["input_ids"] #tokenizer在main里面应该有
    # label = _process_response('')
    bos_token_needed = tokenizer.bos_token is not None
    full_sequence = [[tokenizer.bos_token_id]] if bos_token_needed else []

    full_sequence += [
        dialog_act,
        context,
        [tokenizer.eos_token_id]
    ]

    full_sequence = list(itertools.chain.from_iterable(full_sequence))

    sequences.append(full_sequence)
    # labels.append(label)
    
    # input_ids = sequences

    return_dict = {
    "input_ids": full_sequence,
    }
    
    dataset = Dataset.from_dict(new_dict)
    

    updated_dataset = dataset.map(
    lambda example: return_dict,
    batched=False,
    load_from_cache_file=False
    )
    return updated_dataset

#__function for dialog inpainting__

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
# /cluster/scratch/wangjun/local_data/book_dataset_v4/social_sciences/psychology_2e.json
# /cluster/scratch/wangjun/local_data/book_dataset_v4/science/physics.json
with open('/cluster/scratch/wangjun/local_data/book_dataset_v4/science/physics.json') as f:
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

def extract_keywords(sentences, num_keywords):
    r = Rake()  # Uses stopwords for English from NLTK, and all punctuation characters.
    keywords = []
    
    for sentence in sentences:
        # Remove numbers
        sentence = re.sub(r'\b\d+\b', '', sentence)
        r.extract_keywords_from_text(sentence)
        extracted_keywords = r.get_ranked_phrases()[:num_keywords]  # To get keyword phrases ranked highest to lowest.
        keywords.extend(extracted_keywords)
    
    return keywords
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
# Load dataset from the hub and get a sample
# /cluster/scratch/wangjun/local_data/book_dataset_v4/business/business_ethics.json
# /cluster/scratch/wangjun/local_data/book_dataset_v4/math/algebra_and_trigonometry.json
# with open('/cluster/scratch/wangjun/local_data/book_dataset_v4/math/algebra_and_trigonometry.json') as f:
#     dataset = json.load(f)

# create the DeepSegment instance
# segmenter = DeepSegment('en')
# nlp = stanza.Pipeline(lang='en', processors='tokenize')

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
                keywords = extract_keywords(sentences,2)
                # Generate dialog inpainting
                for idx, sentence in enumerate(sentences):
                    if idx == 0:
                        test_dataset = generate_partial_dialog([sentence], document_title,keywords)
                    else:
                        test_dataset = generate_partial_dialog(dialog + [sentence], document_title, keywords)
                    
                    # print(test_dataset['input_ids'][0])
                    input_ids_tensor = torch.tensor(test_dataset['input_ids'][0])
                    # add repetition penalty
                    results = model.generate(input_ids=input_ids_tensor.unsqueeze(0).cuda(),do_sample=True, top_p=0.9, repetition_penalty=1.5)
                    prediction = tokenizer.decode(results[0].detach().cpu().numpy(), skip_special_tokens=True) #peft prediction
                    
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
                    "utterances": dialog,
                    "keywords": keywords
                }
                with open(prediction_output_file, 'a') as f:
                        json.dump(output_data, f, cls=NumpyEncoder)
