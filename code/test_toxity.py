import requests
import json

def calculate_toxicity(question):
    api_key = 'AIzaSyBnW_3WH0jFDUUEfGKwuTyDDans2KMEC8E'
    """Calculates the toxicity of a question using the Perspective API.

    Args:
        question (str): The question to evaluate.
        api_key (str): Your Perspective API key.

    Returns:
        float: The toxicity score of the question.
    """
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

print(calculate_toxicity('you are such an ugly person!','AIzaSyBnW_3WH0jFDUUEfGKwuTyDDans2KMEC8E'))  # 0-1