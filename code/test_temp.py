# /cluster/scratch/wangjun/local_data/human_eval/alessandro_modify.xlsx
# import pandas as pd


import pandas as pd
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForSequenceClassification
import difflib
import nltk
# from transformers import pipeline, AutoTokenizer, 
import torch

def Q_A_eval2(question: str, given_answer: str, sentences: list):
    if not question:
        print('Empty question was given as input.')
        return 0

    qa_model = pipeline('question-answering', model = 'deepset/deberta-v3-large-squad2')
    nli_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli')
    
    context = ' '.join(sentences)

    output = qa_model(question=question, context=context)

    predicted_answer = output['answer']

    premise = given_answer
    hypothesis = predicted_answer

    inputs = tokenizer.encode_plus(premise, hypothesis, return_tensors='pt')

    outputs = nli_model(**inputs)[0]

    probs = torch.nn.functional.softmax(outputs, dim=-1)

    entailment_score = probs[0,2].item()

    return entailment_score, predicted_answer
def sentences_match(sentence1: str, sentence2: str) -> float:
    # Tokenize both sentences
    tokens1 = nltk.word_tokenize(sentence1)
    tokens2 = nltk.word_tokenize(sentence2)

    # Initialize a SequenceMatcher with the tokens
    sequence_matcher = difflib.SequenceMatcher(None, tokens1, tokens2)

    # Get the similarity ratio
    similarity_score = sequence_matcher.ratio()

    return similarity_score
def Q_A_eval3_deberta(question: str, given_answer: str, sentences: list):
    # Initialize the QA model
    if not question:
        print('Empty question was given as input.')
        return 0, None
    # if not is_question_answerable(sentences,question): #not need another model to see whether answerable or not
    #     return 0

    qa_model = pipeline('question-answering', model = 'deepset/deberta-v3-large-squad2')
    nli_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli')
    
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
    inputs = tokenizer.encode_plus(premise, hypothesis, return_tensors='pt')

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

# Load your xlsx file into a pandas DataFrame
df = pd.read_excel('/cluster/scratch/wangjun/local_data/human_eval/alessandro_modify.xlsx')
# df = pd.read_xlsx(')

# Apply your function to each row and store the results in the new column 'Q_A_eval'
# df['Q_A_eval'] = df.apply(lambda row: Q_A_eval2(row['Sub-questions_2'], row['answer'], row['MWP'].split()), axis=1)

# # Save the updated dataframe to a new xlsx file
# df.to_excel('/cluster/scratch/wangjun/local_data/human_eval/output_file.xlsx', index=False)
df[['Q_A_eval3', 'predicted_answer']] = df.apply(lambda row: pd.Series(Q_A_eval3_deberta(row['Sub-questions_2'], row['answer'], row['MWP'].split())), axis=1)

# Save the updated dataframe to a new xlsx file
df.to_excel('output_file_7_24.xlsx', index=False)