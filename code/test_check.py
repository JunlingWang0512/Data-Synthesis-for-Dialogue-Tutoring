# from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
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
# from transformers import pipeline
import difflib
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForSequenceClassification,AutoModel, GPT2LMHeadModel, GPT2Tokenizer
# import difflib
# import nltk
from ast import literal_eval
# from transformers import pipeline, AutoTokenizer, 
import torch
# import pandas as pd


# tokenizer_qa = AutoTokenizer.from_pretrained('deepset/deberta-v3-large-squad2')
qa_model = pipeline('question-answering', model = 'deepset/deberta-v3-large-squad2')

# nli_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli')
# nli_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
nli_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
def Q_A_eval(question: str, given_answer: str, sentences: list):
    # Initialize the QA model
    if not question:
        print('Empty question was given as input.')
        return 0

    model = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

    # Variables to store the scores and corresponding sentences of the two most likely answers
    top_scores = [0, 0]
    top_sentences = ["", ""]

    # Loop through each sentence in the input sentences
    for sentence in sentences:
        # Use the model to predict the answer from the current sentence
        output = model(question=question, context=sentence)

        # If the score of the current sentence is higher than the top score
        if output['score'] > top_scores[0]:
            # Shift the top score and sentence to the second place
            top_scores[1] = top_scores[0]
            top_sentences[1] = top_sentences[0]

            # Set the current score and sentence as the top ones
            top_scores[0] = output['score']
            top_sentences[0] = sentence

        # If the score of the current sentence is lower than the top score but higher than the second top score
        elif output['score'] > top_scores[1]:
            # Set the current score and sentence as the second top ones
            top_scores[1] = output['score']
            top_sentences[1] = sentence

    # Loop through the two top sentences
    for i in range(2):
        # Split the given answer and the current sentence into words
        given_answer_words = given_answer.split(" ")
        top_sentence_words = top_sentences[i].split(" ")

        # Calculate the similarity between the given answer and the current sentence
        similarity = difflib.SequenceMatcher(None, given_answer_words, top_sentence_words).ratio()

        # If the score of the current sentence is not zero and its similarity to the given answer is 0.8 or higher
        if top_scores[i] != 0 and similarity >= 0.8:
            # Return 1 for the top sentence and 0.5 for the second top sentence
            return 1 if i == 0 else 0.5

    # If none of the top sentences met the conditions, return 0
    return 0
def calculate_toxicity(question):
    time.sleep(0.7) #1 没问题，尝试0.7 0.8.。。。
    api_key = 'AIzaSyBnW_3WH0jFDUUEfGKwuTyDDans2KMEC8E'
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    querystring = {"key": api_key}
    payload = {
        "comment": {"text": question},
        "languages": ["en"],
        "requestedAttributes": {"TOXICITY": {}}
    }
    headers = {
        'Content-Type': 'application/json'
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.request(
                "POST", 
                url, 
                headers=headers, 
                params=querystring, 
                data=json.dumps(payload)
            )

            response_dict = json.loads(response.text)
            toxicity_score = response_dict["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            return toxicity_score
        except Exception as e:
            if attempt < max_retries - 1:  # If it's not the final attempt
                print(f"Error on attempt {attempt + 1} of {max_retries}. Retrying in 1 second...")
                time.sleep(1)  # Wait for 1 second before retrying
                continue
            else:  # If it's the final attempt
                print(f"Error on final attempt. Unable to calculate toxicity. Error: {e}")
                return None




def get_trigrams(text):
    """Extracts all trigrams from the input text.

    Args:
        text (str): The input text.

    Returns:
        list: The list of trigrams in the text.
    """
    tokens = nltk.word_tokenize(text)
    trigrams = list(nltk.trigrams(tokens))
    return trigrams

def distinct_3(text):
    """Calculates the number of distinct 3-grams in the input text.

    Args:
        text (str): The input text.

    Returns:
        int: The number of distinct 3-grams in the text.
    """
    trigrams = get_trigrams(text)
    num_unique_trigrams = len(Counter(trigrams))
    return num_unique_trigrams

def calculate_bigram_entropy(dialog):
    # Tokenize sentences and generate bigrams
    bigrams = []
    for sentence in dialog:
        tokens = nltk.word_tokenize(sentence)
        bigrams.extend(list(ngrams(tokens, 2, pad_left=True, pad_right=True)))

    # Calculate frequencies of bigrams
    freq_dist = FreqDist(bigrams)

    # Calculate probabilities
    probabilities = [freq_dist[bigram] / len(bigrams) for bigram in freq_dist]

    # Compute entropy
    bigram_entropy = entropy(probabilities, base=2)

    return bigram_entropy

openai.api_key = "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"



def extract_score(text):
    match = re.search('Score:\s*(\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1))
    else:
        return None
def generate_response0(prompt,model):
    #intruduction of the task
    # messeage_head = "Task: Generate a high-quality student question that corresponds to the teacher response. I will provide you with a sentence as the teacher's response, which is the answer to the student's question.\n\nInstructions:\n\n1 Your generated student question should be related to the content of the sentence and demonstrate an understanding of the concept or topic discussed in the sentence.\n2 The teacher's response should be unchanged, just put it into json\n3 Use the following JSON format for the dialogue:\n{\"dialogues\": [{\"speaker\": \"student\",\"text\": \"\"},{ \"speaker\": \"teacher\",\"text\": \"\"}]}\n4 When generating a student question, try to put yourself in the shoes of someone who is genuinely curious and wants to learn more about the topic.\n5 Be creative and try to generate a question that a curious student might ask in a real classroom setting.\n6 Only output one JSON variable, do not output anything else!\n7 the teacher's response is: "
    #define the format of the output
    # messeage_end = '\n\nYou should generate one question for one sentence, and the sentence should be the answer for your question. Your question should be high quality and like human. Generate question in the following JSON format, the sentence should be in teacher place, while your question should be in student place:{  "dialogues": [  {  "speaker": "student",  "text": ""  },  {  "speaker": "teacher",  "text": ""  },...  ]}'
    #combine the introduction and the prompt into the input
    messeage_content = prompt

    # print(messeage_content)
    completion = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo", 
        model=model,
        messages=[{"role": "user", "content": messeage_content}]
    )

    return completion
def calculate_coherence_score(dialog):
    
    dialog_str = "\n".join([f"Teacher asks: {dialog[i]}\nStudent answers: {dialog[i+1]}" for i in range(0, len(dialog), 2)])
    
    prompt = f"Assuming you are a linguistic expert, your task is to assess the coherence of the following dialogue generated by an AI system. This dialogue is structured as a teacher-student interaction centered around teaching a passage from a textbook, only the teacher part is generated by AI. When we say 'coherence', we're asking: Does the dialogue logically flow and stay consistent with the conversation's context? Does the AI-generated dialogue maintain the thematic context from beginning to end? Please provide a coherence score from 0 to 10, where 0 signifies no coherence and 10 denotes excellent coherence. The dialog is: \n\n{dialog_str}\n\n Please provide me with the score only in your response in this format: Score:<number>"
    # print('prompt:',prompt)
    # print('__________________________generated______________________________________________')
    # inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
    # outputs = model.generate(inputs, max_length=512, do_sample=True)
    result = generate_response0(prompt,'gpt-4-0314')
    input_string = result['choices'][0]['message']['content']
    
    return extract_score(input_string)

def calculate_correctness_score(dialog):
    
    dialog_str = "\n".join([f"Teacher asks: {dialog[i]}\nStudent answers: {dialog[i+1]}" for i in range(0, len(dialog), 2)])
    
    prompt = f"Assuming you are a linguistic expert, your task is to assess the correctness of the following dialogue generated by an AI system. This dialogue is structured as a teacher-student interaction centered around teaching a passage from a textbook, only the teacher part is generated by AI. When we say 'correctness', we're asking: Does the dialogue provide accurate and factual question for each response? \n\n Please provide a coherence score from 0 to 10, where 0 signifies all questions are not correct and 10 denotes excellent correctness. The dialog is: \n\n{dialog_str}\n\n Please provide me with the score only in your response in this format: Score:<number>"
    # print('prompt:',prompt)
    # print('__________________________generated______________________________________________')
    # inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
    # outputs = model.generate(inputs, max_length=512, do_sample=True)
    result = generate_response0(prompt,'gpt-4-0314')
    input_string = result['choices'][0]['message']['content']
    extract_score(input_string)
    return extract_score(input_string)
# import language_tool_python

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

def is_general_question(question, similarity_threshold=0.9):
    general_patterns = [
        r'\bwhat\b.*\belse is discussed\b',  # Matches 'what else...is discussed'
        r'\bwhat else\b.*\bdocument said\b',  # Matches 'what else...document said'
        # r'\bwhat\b.*\bimportant\b',  # Matches 'what...important'
        # r'\bwhat\b.*\btell\b',  # Matches 'what...tell'
        # r'\bwhat\b.*\bthink\b',  # Matches 'what...think'
        # r'\bare\b.*\bany other\b',  # Matches 'are...any other'
        # r'\bwhat\b.*\bmain\b',  # Matches 'what...main'
        r'\bwhat else\b.*\babout the article\b'  # Matches 'what else...about the article'
        # r'\bwhat\b.*\bin\b',  # Matches 'what...in'
        # r'\bwhat\b.*\bone\b',  # Matches 'what...one'
        # r'\bwhat\b.*\bpurpose\b',  # Matches 'what...purpose'
        # r'\bwhat\b.*\bfirst\b',  # Matches 'what...first'
        # r'\bdid\b.*\bany other\b',  # Matches 'did...any other'
    ]
    question_list = [
    "What do you think we're going to learn?",
    "Are there any other important things we need to know about this lesson?",
    "Are there any other points we should know about this lesson?",
    'Are there any other examples?',
    'What is the most important part of this topic to you?',
    "Why is that?",
    "What else is significant?",
    "What else are you studying?",
    "What else can you tell me about the document?",
    "What else can you tell me about the article?",
    "What did they find?",
    "What else was said?",
    'What else does the text discuss?',
    "What else is interesting?",
    "What is your main takeaway from this?",
    "What other information did you find helpful?",
    "what is the first topic of the material?",
    "why did they do this?",
    "Why does it say this?",
    "What else can you tell me?",
    "Can you tell me more about this?",
    "What's in it?",
    "What is one of the things you have to do before this?",
    "What else did you need to do before this?",
    "What is the purpose of this study material?",
    "What else does this study material include?",
    "Are there any other interesting facts in this article?",
    "Are there any notable topics covered in this study material?",
    "Are there any other interesting topics or facts discussed?",
    "What else did you find?",
    "What are the main points in the study material?",
    "What else is important in the material?",
    "what else is important in this section?",
    "what was so important about this?",
    "What other things do you find important?",
    "What else did you find interesting in the section?",
    "What else is important?",
    "What was the earliest thing you learned about this paper?",
    "Did he state any other important aspects of this paper?",
    "what's the most important thing that you want to know about this paper?",
    "Why did you decide to write this document?"
]


    question = question.lower()

    for pattern in general_patterns:
        if re.search(pattern, question):
            return 1
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer().fit(question_list + [question])

    # Vectorize the question list and the input question
    question_list_vec = vectorizer.transform(question_list)
    question_vec = vectorizer.transform([question])

    # Calculate cosine similarities
    similarities = cosine_similarity(question_vec, question_list_vec)

    # If the maximum similarity is greater than the threshold, return True
    if similarities.max() > similarity_threshold:
        return 1

    return 0

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

def sentences_match(sentence1: str, sentence2: str) -> float:
    # Tokenize both sentences
    tokens1 = nltk.word_tokenize(sentence1)
    tokens2 = nltk.word_tokenize(sentence2)

    # Initialize a SequenceMatcher with the tokens
    sequence_matcher = difflib.SequenceMatcher(None, tokens1, tokens2)

    # Get the similarity ratio
    similarity_score = sequence_matcher.ratio()

    return similarity_score

def Q_A_eval4(question: str, given_answer: str, sentences: list):
    # Initialize the QA model
    repetition = sentences_match(question,given_answer)
    if not question:
        print('Empty question was given as input.')
        return 0,"",0,repetition
    # if not is_question_answerable(sentences,question): #not need another model to see whether answerable or not
    #     return 0

    # qa_model = pipeline('question-answering', model = 'deepset/deberta-v3-large-squad2')
    # nli_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
    # nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
    
    # Join all sentences into a single context
    context = ' '.join(sentences)

    

    # Use the QA model to predict the answer from the context
    output = qa_model(question=question, context=context)
    print('qa_output',output)
    # Store the predicted answer
    predicted_answer = output['answer']

    if not predicted_answer.strip():  #if cannot predict any answer, return 0 #or output['score'] < some_threshold:
        return 0,predicted_answer,0
    # Check the token-level similarity
    similarity_score = sentences_match(given_answer, predicted_answer)
    if similarity_score > 0.8:
        return 1,predicted_answer,similarity_score,repetition

    # Prepare the inputs for the NLI model
    premise = given_answer
    hypothesis = predicted_answer

    # Encode the inputs
    inputs = nli_tokenizer.encode_plus(premise, hypothesis, return_tensors='pt')

    # Get the model's predictions
    outputs = nli_model(**inputs)[0]

    # Get the probabilities by applying the softmax function
    probs = torch.nn.functional.softmax(outputs, dim=-1)

    # Get the max probability's index (0: contradiction, 1: neutral, 2: entailment)
    max_index = torch.argmax(probs).item()

    # Check the prediction and return the appropriate score
    entailment_similarity_threshold = 0.4
    neutral_similarity_threshold = 0.05 #since we find similarity cannot directly reflect the quality of results,we say if there is acutally 5% overlap, it should have some factual consistence, we assign 0.5,which conresponding to human judgement.
    if max_index == 2:  # entailment
        print('entailment')
        if similarity_score >= entailment_similarity_threshold:
            return 1,predicted_answer,similarity_score,repetition
        else:
            return 0.5 , predicted_answer,similarity_score,repetition
        # return 1,predicted_answer
    elif max_index == 1:  # neutral
        if similarity_score >= neutral_similarity_threshold:
            return 0.5,predicted_answer,similarity_score,repetition
        else:
            return 0,predicted_answer,similarity_score,repetition
    elif max_index == 0:  # contradiction
        return 0,predicted_answer,similarity_score,repetition
    else:
        return 0,predicted_answer,similarity_score,repetition  # this should never happen
#############################################################################################
file_path = '/cluster/scratch/wangjun/GPT_results/GPT_3.5/math_search_output.json'

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
for obj in selected_data:
    dialog = obj['utterances']
    sentences = obj['sentences'] #used by q&a prediction model
    count += 1
    print('dialog count = ', count)
    # print(dialog)
    if dialog == []:continue
    tuples = [dialog[i:i+2] for i in range(0, len(dialog), 2)]
    for tuple_ in tuples:
        score_dict = {}
        score_dict['tuple'] = tuple_
        # score_dict['relevance'] = relevance_score(tuple_)
        # score_dict['flesch_reading_ease'] = flesch_reading_ease(tuple_)
        # score_dict['bigram_entropy'] = calculate_bigram_entropy(tuple_)
        # score_dict['is_general'] = is_general_question(tuple_[0])
        # score_dict['distinct_3'] = distinct_3(tuple_[0])
        # score_dict['toxicity'] = calculate_toxicity(tuple_[0])
        score_dict['Q_A_eval4'],_1,_2,_3 = Q_A_eval4(tuple_[0],tuple_[1],sentences)
        # score_dict['correctness'] = calculate_correctness_score(tuple_)
        # score_dict['coherence'] = calculate_coherence_score(tuple_)
        # Append the dictionary to the score_data list
        score_data.append(score_dict)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(score_data)

# Calculate the average scores and append them to the DataFrame
# df['toxicity'] = df['toxicity'].replace({None: pd.NA}).astype(float)  # replace None with NaN and ensure data type is float
# avg_scores = df[['relevance', 'flesch_reading_ease', 'bigram_entropy', 'is_general', 'distinct_3','Q_A_eval']].mean(skipna=True)
# df.loc['Average'] = ['Average'] + avg_scores.tolist()
avg_scores = df[['Q_A_eval4']].mean(skipna=True)
df.loc['Average'] = ['Average'] + avg_scores.tolist()
print(df)
df.to_csv('7_27_qa_eval_gpt3.5_math.csv')