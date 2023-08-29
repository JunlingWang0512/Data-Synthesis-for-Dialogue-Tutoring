import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import json


# def textrank_central_score(sentences):
#     # Compute the TF-IDF matrix for the sentences
#     vectorizer = TfidfVectorizer().fit_transform(sentences)
#     vectors = vectorizer.toarray()
    
#     # Compute the cosine similarity matrix
#     cosine_matrix = cosine_similarity(vectors)
    
#     # Build the graph with sentences as nodes and cosine similarity scores as edges
#     nx_graph = nx.from_numpy_array(cosine_matrix)
    
#     # Compute the PageRank scores for each sentence
#     scores = nx.pagerank(nx_graph)
    
#     return scores

# # Test the function with a list of sentences
# test_sentences = [
#     "To formalize our work, we will begin by drawing angles on an x-y coordinate plane.",
#     "Angles can occur in any position on the coordinate plane, but for the purpose of comparison, the convention is to illustrate them in the same position whenever possible.",
#     "An angle is in standard position if its vertex is located at the origin, and its initial side extends along the positive x-axis.",
#     "See Figure 5."
#   ]

# textrank_scores_list_input = textrank_central_score(test_sentences)
# print('textrank_scores_list_input',textrank_scores_list_input)


from sklearn.decomposition import TruncatedSVD

def lsa_central_score(sentences):
    # Compute the TF-IDF matrix for the sentences
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    
    # Apply Singular Value Decomposition (SVD)
    svd = TruncatedSVD(n_components=1, random_state=42)
    lsa_embeddings = svd.fit_transform(vectorizer)
    
    # Score sentences based on the magnitude of the vectors in the reduced space
    scores = {i: np.sum(lsa_embeddings[i]**2) for i in range(len(sentences))}
    
    return scores

# # Test the function with the list of sentences
# lsa_scores = lsa_central_score(test_sentences)
# # lsa_scores
# print('lsa_socres',lsa_scores)


############################# test_filtering_data_processing
def filter_utterances(data, threshold=0.1):
    """Filter only the utterances based on LSA score of the corresponding sentences, 
    considering the relationship between sentences and utterances."""
    modified_data = []
    
    for section in data:
        sentences = section["sentences"]
        utterances = section["utterances"].copy()  # Copy to avoid modifying original
        
        # Calculate LSA scores
        lsa_scores = lsa_central_score(sentences)
        
        # Filter utterances based on the threshold of corresponding sentences
        filtered_utterances = []
        for i, sentence in enumerate(sentences):
            if lsa_scores[i] >= threshold:
                # Add the sentence above the current one and the sentence itself
                filtered_utterances.extend(utterances[2*i : 2*i+2])
        
        # Modify only the utterances of the section
        section["utterances"] = filtered_utterances
        with open('/cluster/scratch/wangjun/filtering_result/business_student_filter.json', 'a') as f:
            
            json.dump(section, f)
            # processed_data = []
        modified_data.append(section)
    
    return modified_data

# Process the sample data again using the corrected function
data = []
processed_data = []  # new list for processed dialogues
decoder = json.JSONDecoder()

with open('/cluster/scratch/wangjun/Student_question_result/business_student_search_output.json', 'r') as f:
    text = f.read()
    while text:
        obj, idx = decoder.raw_decode(text)
        data.append(obj)
        text = text[idx:].lstrip()
        
modified_json_data_corrected = filter_utterances(data)

# Let's look at the modified utterances of the first section
# modified_json_data_corrected[0]["utterances"]
