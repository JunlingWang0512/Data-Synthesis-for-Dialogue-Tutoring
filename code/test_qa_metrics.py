from transformers import pipeline, AutoModelForSeq2SeqLM,AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForSequenceClassification
import difflib
import nltk
from ast import literal_eval
# from transformers import pipeline, AutoTokenizer, 
import torch
import pandas as pd

import spacy



# tokenizer_qa = AutoTokenizer.from_pretrained('deepset/deberta-v3-large-squad2')
qa_model = pipeline('question-answering', model = 'deepset/deberta-v3-large-squad2')

nli_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli')
nli_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')


# qa_model = pipeline('question-answering', model = 'deepset/deberta-v3-large-squad2')
    # nli_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
    # tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli')
#VERSION 1
# def Q_A_eval(question: str, given_answer: str, sentences: list):
#     # Initialize the QA model
#     if not question:
#         print('Empty question was given as input.')
#         return 0

#     model = pipeline("question-answering", model='deepset/deberta-v3-large-squad2')
#     # deepset/deberta-v3-large-squad2
#     # distilbert-base-cased-distilled-squad
   

#     # Variables to store the scores and corresponding sentences of the two most likely answers
#     top_scores = [0, 0]
#     top_sentences = ["", ""]

#     # Loop through each sentence in the input sentences
#     for sentence in sentences:
#         # Use the model to predict the answer from the current sentence
#         output = model(question=question, context=sentence)

#         # If the score of the current sentence is higher than the top score
#         if output['score'] > top_scores[0]:
#             # Shift the top score and sentence to the second place
#             top_scores[1] = top_scores[0]
#             top_sentences[1] = top_sentences[0]

#             # Set the current score and sentence as the top ones
#             top_scores[0] = output['score']
#             top_sentences[0] = sentence

#         # If the score of the current sentence is lower than the top score but higher than the second top score
#         elif output['score'] > top_scores[1]:
#             # Set the current score and sentence as the second top ones
#             top_scores[1] = output['score']
#             top_sentences[1] = sentence

#     # Loop through the two top sentences
#     for i in range(2):
#         # Split the given answer and the current sentence into words
#         given_answer_words = given_answer.split(" ")
#         top_sentence_words = top_sentences[i].split(" ")

#         # Calculate the similarity between the given answer and the current sentence
#         similarity = difflib.SequenceMatcher(None, given_answer_words, top_sentence_words).ratio()
#         # use deberta-large-mnli instead

#         # If the score of the current sentence is not zero and its similarity to the given answer is 0.8 or higher
#         if top_scores[i] != 0 and similarity >= 0.8:
#             # Return 1 for the top sentence and 0.5 for the second top sentence
#             return 1 if i == 0 else 0.5

#     # If none of the top sentences met the conditions, return 0
#     return 0
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import torch



#VERSION 2
# def Q_A_eval2(question: str, given_answer: str, sentences: list):
#     # Initialize the QA model
#     if not question:
#         print('Empty question was given as input.')
#         return 0

#     qa_model = pipeline('question-answering', model = 'deepset/deberta-v3-large-squad2')
#     nli_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
#     tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli')
    
#     # Join all sentences into a single context
#     context = ' '.join(sentences)

#     # Use the QA model to predict the answer from the context
#     output = qa_model(question=question, context=context)
#     print('qa_output',output)
#     # Store the predicted answer
#     predicted_answer = output['answer']

#     # Prepare the inputs for the NLI model
#     premise = given_answer
#     hypothesis = predicted_answer

#     # Encode the inputs
#     inputs = tokenizer.encode_plus(premise, hypothesis, return_tensors='pt')

#     # Get the model's predictions
#     outputs = nli_model(**inputs)[0]

    
#     # Get the probabilities by applying the softmax function
#     probs = torch.nn.functional.softmax(outputs, dim=-1)

#     # Get the similarity score for entailment
#     # similarity_score = probs[0,0].item()
#     # contradiction_score = probs[0,0].item()
#     # neutral_score = probs[0,1].item()
#     entailment_score = probs[0,2].item()

#     return entailment_score





def sentences_match(sentence1: str, sentence2: str) -> float:
    # Tokenize both sentences
    tokens1 = nltk.word_tokenize(sentence1)
    tokens2 = nltk.word_tokenize(sentence2)

    # Initialize a SequenceMatcher with the tokens
    sequence_matcher = difflib.SequenceMatcher(None, tokens1, tokens2)

    # Get the similarity ratio
    similarity_score = sequence_matcher.ratio()

    return similarity_score


def is_question_answerable(document: list, question: str) -> bool:
    # Join the list of sentences into a single string
    document = ' '.join(document)
    model_name = "google/electra-large-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    inputs = tokenizer.encode_plus(question, document, return_tensors='pt')
    outputs = model(**inputs)

    # The model returns a tuple with start and end logits
    # We can use argmax to find the position in the input tokens with the highest score
    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item()

    # If the model can't find an answer, it will return the position of the [CLS] token, which is 0
    if answer_start == 0 and answer_end == 0:
        return False

    return True


#VERSION 3
def Q_A_eval3_deberta(question: str, given_answer: str, sentences: list):
    # Initialize the QA model
    if not question:
        print('Empty question was given as input.')
        return 0, None
    # if not is_question_answerable(sentences,question): #not need another model to see whether answerable or not
    #     return 0

    # qa_model = pipeline('question-answering', model = 'deepset/deberta-v3-large-squad2')
    # nli_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
    # tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli')
    
    # Join all sentences into a single context
    context = ' '.join(sentences)

    # Use the QA model to predict the answer from the context
    output = qa_model(question=question, context=context)
    print('qa_output',output)
    # Store the predicted answer
    predicted_answer = output['answer']

    if not predicted_answer.strip():  #if cannot predict any answer, return 0 #or output['score'] < some_threshold:
        return 0,predicted_answer
    # Check the token-level similarity
    similarity_score = sentences_match(given_answer, predicted_answer)
    if similarity_score > 0.8:
        return 1,predicted_answer

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
    if max_index == 2:  # entailment
        print('entailment')
        return 1,predicted_answer
    elif max_index == 1:  # neutral
        return similarity_score,predicted_answer
    elif max_index == 0:  # contradiction
        return 0,predicted_answer
    else:
        return None,predicted_answer  # this should never happen

#VERSION 3
def Q_A_eval3_roberta(question: str, given_answer: str, sentences: list):
    # Initialize the QA model
    if not question:
        print('Empty question was given as input.')
        return 0,predicted_answer
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
        return 0,predicted_answer
    # Check the token-level similarity
    similarity_score = sentences_match(given_answer, predicted_answer)
    if similarity_score > 0.8:
        return 1,predicted_answer

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
    if max_index == 2:  # entailment
        print('entailment')
        return 1,predicted_answer
    elif max_index == 1:  # neutral
        return similarity_score,predicted_answer
    elif max_index == 0:  # contradiction
        return 0,predicted_answer
    else:
        return None,predicted_answer  # this should never happen

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




def Q_A_eval5_entailment_score(question: str, given_answer: str, sentences: list):
    # Initialize the QA model
    repetition = sentences_match(question,given_answer)
    if not question:
        print('Empty question was given as input.')
        return 0,"",0,repetition

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

    # Compute probability of entailment and non-entailment
    entailment_prob = probs[0, 2].item()
    non_entailment_prob = probs[0, 0].item() + probs[0, 1].item()

    return entailment_prob,predicted_answer,similarity_score,repetition

nlp = spacy.load('en_core_web_sm')

def is_span_in_sentence(sentence, span):
    # Tokenize the sentence into a list of sentences
    sentences = list(nlp(sentence).sents)
    
    # Check if the span is in any of the sentences
    for sent in sentences:
        if span in sent.text:
            return True
    return False

def Q_A_eval6_span_binary(question: str, given_answer: str, sentences: list):
    # Initialize the QA model
    repetition = sentences_match(question,given_answer)
    if not question:
        print('Empty question was given as input.')
        return 0,"",0,repetition

    # Join all sentences into a single context
    context = ' '.join(sentences)

    # Use the QA model to predict the answer from the context
    output = qa_model(question=question, context=context)
    print('qa_output',output)
    # Store the predicted answer
    predicted_answer = output['answer']

    if predicted_answer.strip() == '[CLS]':  #if cannot predict any answer, return 0 #or output['score'] < some_threshold:
        return 0,predicted_answer,0
    # Check the token-level similarity
    similarity_score = sentences_match(given_answer, predicted_answer)
    if similarity_score > 0.8:
        return 1,predicted_answer,similarity_score,repetition

    if is_span_in_sentence(given_answer,predicted_answer):
        return 1,predicted_answer,similarity_score,repetition
    else:
        return 0,predicted_answer,similarity_score,repetition



def Q_A_eval7_generative_model(question: str, given_answer: str, sentences: list):
    # Initialize the FLAN T5 base model
    flan_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    flan_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

    repetition = sentences_match(question,given_answer)
    if not question:
        print('Empty question was given as input.')
        return 0,"",0,repetition

    # Join all sentences into a single context
    context = ' '.join(sentences)

    # Prepare the inputs for the Flan T5 base model
    instruction = "Given a passage of text and a question, generate an answer in natural language or say \"I don't know\" if the question is unanswerable."
    prompt = f"{instruction}\n\nPassage: {context}\nQuestion: {question}"
    inputs = flan_tokenizer.encode(prompt, return_tensors='pt')

    # Generate an output sequence (answer) with the Flan T5 base model
    outputs = flan_model.generate(inputs, max_length=200, temperature=0.7)
    predicted_answer = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

    print('predicted_answer',predicted_answer)

    if not predicted_answer.strip():  #if cannot predict any answer, return 0 #or output['score'] < some_threshold:
        return 0,predicted_answer,0

    # Check the token-level similarity
    similarity_score = sentences_match(given_answer, predicted_answer)

    return similarity_score,predicted_answer,similarity_score,repetition


def Q_A_eval8_question_score(question: str, given_answer: str, sentences: list):
    # Initialize the QA model
    repetition = sentences_match(question,given_answer)
    if not question:
        print('Empty question was given as input.')
        return 0,"",0,repetition

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

    question_prob = output['score']
    # # Prepare the inputs for the NLI model
    # premise = given_answer
    # hypothesis = predicted_answer

    # # Encode the inputs
    # inputs = nli_tokenizer.encode_plus(premise, hypothesis, return_tensors='pt')

    # # Get the model's predictions
    # outputs = nli_model(**inputs)[0]

    # # Get the probabilities by applying the softmax function
    # probs = torch.nn.functional.softmax(outputs, dim=-1)

    # # Compute probability of entailment and non-entailment
    # entailment_prob = probs[0, 2].item()
    # non_entailment_prob = probs[0, 0].item() + probs[0, 1].item()

    return question_prob,predicted_answer,similarity_score,repetition
# example usage
# question = "What is the reference angle for the first quadrant?"
# given_answer = "An angle in the first quadrant is its own reference angle."
# sentences = [
#     "An angle in the first quadrant is its own reference angle.",
#     "For an angle in the second or third quadrant, the reference angle is $$\\(\\left| {\\pi - t} \\right|\\)$$ or $$\\(\\left| {180{^\\circ} - t} \\right|.\\)$$",
#     "For an angle in the fourth quadrant, the reference angle is $$\\(2\\pi - t\\)$$ or $$\\(360{^\\circ} - t.\\)$$",
#     "If an angle is less than $$\\(0\\)$$ or greater than $$\\(2\\pi,\\)$$ add or subtract $$\\(2\\pi\\)$$ as many times as needed to find an equivalent angle between $$\\(0\\)$$ and $$\\(2\\pi.\\)$$"
#   ]
# # print("is_question_answerable",is_question_answerable(sentences, question))

# score = Q_A_eval3_roberta(question, given_answer, sentences)
# print("q-a-eval",score)


# Function to preprocess the context column
def preprocess_context(context_str):
    # Remove leading/trailing whitespaces and newlines
    context_str = context_str.strip()
    # Replace newline characters within sentences with spaces
    context_str = context_str.replace('\n', ' ')
    return context_str
# 文件级别
# Load your xlsx file into a pandas DataFrame
# /cluster/scratch/wangjun/local_data/human_eval/key+post_human_eval_dataset_7_25.xlsx
# /cluster/scratch/wangjun/local_data/human_eval/GPT3.5-human_eval.xlsx
df = pd.read_excel('/cluster/scratch/wangjun/local_data/human_eval/key+post_human_eval_8_14_100.xlsx')

# Preprocess the 'context' column
df['context'] = df['context'].apply(preprocess_context)

# Convert the 'context' column from a string representation of a list to an actual list of sentences
df['context'] = df['context'].apply(literal_eval)

# Apply your function to each row and store the results in the new columns 'Q_A_eval3' and 'predicted_answer'
df[['QFactScore', 'predicted_answer','similarity_score','repetition']] = df.apply(lambda row: pd.Series(Q_A_eval5_entailment_score(row['question'], row['answer'], row['context'])), axis=1)

# Save the updated dataframe to a new xlsx file
average_score1 = df['QFactScore'].mean()
print('key+post average_qa:',average_score1)
df.to_excel('qfactscore_key+post_human_eval_8_14_100.xlsx', index=False)
##################################################################
# /cluster/scratch/wangjun/local_data/human_eval/GPT3.5-human_eval.xlsx
df = pd.read_excel('/cluster/scratch/wangjun/local_data/human_eval/GPT3.5-human_eval_8_14_100.xlsx')
# Preprocess the 'context' column
df['context'] = df['context'].apply(preprocess_context)

# Convert the 'context' column from a string representation of a list to an actual list of sentences
df['context'] = df['context'].apply(literal_eval)

# Apply your function to each row and store the results in the new columns 'Q_A_eval3' and 'predicted_answer'
df[['QFactScore', 'predicted_answer','similarity_score','repetition']] = df.apply(lambda row: pd.Series(Q_A_eval5_entailment_score(row['question'], row['answer'], row['context'])), axis=1)

# Save the updated dataframe to a new xlsx file
average_score2 = df['QFactScore'].mean()
print('key+post average_QFactScore:',average_score1)
print('gpt-3.5 average_QFactScore:',average_score2)


df.to_excel('qfactscore_GPT3.5-human_eval_8_14_100.xlsx', index=False)
print('done')

