import math
from collections import defaultdict
from typing import List, Tuple
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import json

def word_frequencies(sentences: List[str]) -> defaultdict:
    """Compute the frequency of each word in the passage."""
    frequencies = defaultdict(int)
    
    for sentence in sentences:
        for word in sentence.split():
            word = word.lower().strip(".,!?()")
            frequencies[word] += 1
            
    return frequencies

def sentence_entropy(sentence: str, frequencies: defaultdict, total_words: int) -> float:
    """Compute the entropy of a sentence based on word frequencies."""
    entropy = 0.0
    for word in sentence.split():
        word = word.lower().strip(".,!?()")
        p_w = frequencies[word] / total_words
        entropy -= p_w * math.log(p_w, 2)
        
    return entropy / len(sentence.split())  # Normalize by sentence length

def rank_sentences(sentences: List[str]) -> List[Tuple[str, float]]:
    """Rank sentences based on their entropy."""
    frequencies = word_frequencies(sentences)
    total_words = sum(frequencies.values())
    
    scores = []
    for sentence in sentences:
        scores.append((sentence_entropy(sentence, frequencies, total_words)))
        
    return scores

# Latent Semantic Analysis (LSA) is a technique in natural language processing and information retrieval that reduces the dimensionality of term-document matrices using singular value decomposition (SVD). It can be used for various purposes, including topic modeling and document similarity calculations
def lsa_central_score(sentences):
    # Compute the TF-IDF matrix for the sentences
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    
    # Apply Singular Value Decomposition (SVD)
    svd = TruncatedSVD(n_components=1, random_state=42)
    lsa_embeddings = svd.fit_transform(vectorizer)
    
    # Score sentences based on the magnitude of the vectors in the reduced space
    scores = {i: np.sum(lsa_embeddings[i]**2) for i in range(len(sentences))}
    
    return scores
# Sample usage:
passage = [
    "To formalize our work, we will begin by drawing angles on an x-y coordinate plane.",
    "Angles can occur in any position on the coordinate plane, but for the purpose of comparison, the convention is to illustrate them in the same position whenever possible.",
    "An angle is in standard position if its vertex is located at the origin, and its initial side extends along the positive x-axis.",
    "See Figure 5."
  ]

scores = rank_sentences(passage)
print(scores)

print(lsa_central_score(passage))

# entropy:
# [0.1123468730360788, 0.14959808656775336, 0.1340491342664976, 0.08852933995330681]
# lsa:
# {0: 0.1836564160349811, 1: 0.6486275172839653, 2: 0.5279318950707919, 3: 1.5217794504504082e-30}