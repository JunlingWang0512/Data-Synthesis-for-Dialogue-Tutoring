from transformers import pipeline

def check_similarity(question: str, answer: str):
    # Instantiate a transformer-based model that has been trained on a QA task
    qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

    # Use the model to generate an answer to the question, using the answer as the context
    output = qa_model({'question': question, 'context': answer})

    print(output)
    # If the model's generated answer matches the provided answer, we say the answer is valid
    if output['answer'].lower() == answer.lower():
        return 'The answer is a valid response to the question.'
    else:
        return 'The answer is not a valid response to the question.'

question = "What is the capital of France?"
answer = "The capital of France is Paris."

print(check_similarity(question, answer))
