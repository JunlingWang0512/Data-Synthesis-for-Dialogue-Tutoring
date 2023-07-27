import nltk
from collections import Counter

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

print(distinct_3('what is the capital of paris?'))