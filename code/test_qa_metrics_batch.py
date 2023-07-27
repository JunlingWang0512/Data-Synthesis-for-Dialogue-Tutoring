# qa_eval_batch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForSequenceClassification
import difflib
import nltk
from ast import literal_eval
# from transformers import pipeline, AutoTokenizer, 
import torch
import pandas as pd
# Load the tokenizer and model outside the function
tokenizer_qa = AutoTokenizer.from_pretrained('deepset/deberta-v3-large-squad2')
model_qa = AutoModelForQuestionAnswering.from_pretrained('deepset/deberta-v3-large-squad2')

tokenizer_nli = AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli')
model_nli = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_qa = model_qa.to(device)
model_nli = model_nli.to(device)
# Function to tokenize a batch of contexts and questions
def prepare_inputs(questions, contexts):
    # Join each list of strings into a single string
    contexts = [' '.join(context) for context in contexts]
    return tokenizer_qa(questions, contexts, truncation=True, padding=True, return_tensors='pt')


# Function to get the best answer from the model's outputs
def get_best_answer(start_scores, end_scores, inputs):
    # Get the most probable start and end tokens
    max_start_scores, start_indexes = torch.max(start_scores, dim=1)
    max_end_scores, end_indexes = torch.max(end_scores, dim=1)
    
    # Get the most probable answer spans
    answers = []
    for i in range(len(start_indexes)):
        start_index = start_indexes[i]
        end_index = end_indexes[i]
        answers.append(tokenizer_qa.decode(inputs['input_ids'][i][start_index:end_index+1]))
        
    return answers, max_start_scores

def Q_A_eval_batch(questions, answers, contexts):
    # Prepare the inputs for the model
    inputs = prepare_inputs(questions, contexts)
    
    # Move the inputs to the GPU if available
    if torch.cuda.is_available():
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    # Get the model's outputs
    start_scores, end_scores = model_qa(**inputs).values()
    
    # Get the best answer and its score
    predicted_answers, scores = get_best_answer(start_scores, end_scores, inputs)
    
    results = []
    for question, answer, context, predicted_answer, score in zip(questions, answers, contexts, predicted_answers, scores):
        # if score < 0.01:  # You can adjust this threshold based on your specific use case
        #     results.append((0, None))
        #     continue
            
        # Check the token-level similarity
        similarity_score = difflib.SequenceMatcher(None, nltk.word_tokenize(answer), nltk.word_tokenize(predicted_answer)).ratio()
        if similarity_score > 0.8:
            results.append((1, predicted_answer))
            continue
            
        # Prepare the inputs for the NLI model
        nli_inputs = tokenizer_nli.encode_plus(answer, predicted_answer, return_tensors='pt')
        
        # Move the inputs to the GPU if available
        if torch.cuda.is_available():
            nli_inputs = {name: tensor.to(device) for name, tensor in nli_inputs.items()}

        # Get the model's predictions
        nli_outputs = model_nli(**nli_inputs)
        
        # Get the probabilities by applying the softmax function
        probs = torch.nn.functional.softmax(nli_outputs.logits, dim=-1)
        
        # Get the max probability's index (0: contradiction, 1: neutral, 2: entailment)
        max_index = torch.argmax(probs).item()
        
        # Check the prediction and return the appropriate score
        if max_index == 2:  # entailment
            results.append((1, predicted_answer))
        elif max_index == 1:  # neutral
            results.append((similarity_score, predicted_answer))
        elif max_index == 0:  # contradiction
            results.append((0, predicted_answer))
        else:
            results.append((None, predicted_answer))  # this should never happen
    
    return results


def preprocess_context(context_str):
    # Remove leading/trailing whitespaces and newlines
    context_str = context_str.strip()
    # Replace newline characters within sentences with spaces
    context_str = context_str.replace('\n', ' ')
    return context_str


df = pd.read_excel('/cluster/scratch/wangjun/local_data/human_eval/human_eval_dataset_7_25.xlsx')

# Preprocess the 'context' column
df['context'] = df['context'].apply(preprocess_context)

# Convert the 'context' column from a string representation of a list to an actual list of sentences
df['context'] = df['context'].apply(literal_eval)
# df[['Q_A_eval3', 'predicted_answer']] = pd.DataFrame(Q_A_eval_batch(df['question'], df['answer'], df['context']), index=df.index)
# df[['Q_A_eval3', 'predicted_answer']] = pd.DataFrame(Q_A_eval_batch(df['question'].tolist(), df['answer'].tolist(), df['context'].tolist()), index=df.index)


# # Save the updated dataframe to a new xlsx file
# df.to_excel('/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/output_file_7_25_deberta_batch.xlsx', index=False)
# Function to process batches of data
def process_batches(df, batch_size=5):
    # Initialize empty lists to store results
    results_Q_A_eval3 = []
    results_predicted_answer = []

    # Iterate over the dataframe in batches
    for i in range(0, df.shape[0], batch_size):
        # Get the current batch
        df_batch = df.iloc[i:i+batch_size]
        # Get the questions, answers, and contexts for the current batch
        questions = df_batch['question'].tolist()
        answers = df_batch['answer'].tolist()
        contexts = df_batch['context'].tolist()
        # Evaluate the current batch and append the results
        batch_results = Q_A_eval_batch(questions, answers, contexts)
        for res in batch_results:
            results_Q_A_eval3.append(res[0])
            results_predicted_answer.append(res[1])

    return results_Q_A_eval3, results_predicted_answer

# Process the dataframe in batches and get the results
results_Q_A_eval3, results_predicted_answer = process_batches(df)

# Add the results to the dataframe
df['Q_A_eval3'] = results_Q_A_eval3
df['predicted_answer'] = results_predicted_answer

# Save the updated dataframe to a new xlsx file
df.to_excel('/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/output_file_7_25_deberta_batch.xlsx', index=False)
