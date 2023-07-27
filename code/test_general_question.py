


# print(is_general_question('What else can you tell me about the document?'))  # Outputs: True
# print(is_general_question('Are there any other important things we need to know about this lesson?'))  # Outputs: True
# print(is_general_question("What's the capital of France?"))  # Outputs: False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def is_general_question(question, similarity_threshold=0.9):
    general_patterns = [
        r'\bwhat\b.*\belse\b',  # Matches 'what else'
        r'\bwhat\b.*\bimportant\b',  # Matches 'what...important'
        # r'\bwhat\b.*\btell\b',  # Matches 'what...tell'
        # r'\bwhat\b.*\bthink\b',  # Matches 'what...think'
        r'\bare\b.*\bany other\b',  # Matches 'are...any other'
        r'\bwhat\b.*\bmain\b',  # Matches 'what...main'
        r'\bwhat\b.*\btopic\b',  # Matches 'what...topic'
        # r'\bwhat\b.*\bin\b',  # Matches 'what...in'
        # r'\bwhat\b.*\bone\b',  # Matches 'what...one'
        # r'\bwhat\b.*\bpurpose\b',  # Matches 'what...purpose'
        # r'\bwhat\b.*\bfirst\b',  # Matches 'what...first'
        r'\bdid\b.*\bany other\b',  # Matches 'did...any other'
    ]
    question_list = [
    "What do you think we're going to learn?",
    "Are there any other important things we need to know about this lesson?",
    "Are there any other points we should know about this lesson?",
    "Why is that?",
    "What else is significant?",
    "What else are you studying?",
    "What else can you tell me about the document?",
    "What else can you tell me about the article?",
    "What did they find?",
    "What else was said?",
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
            return True
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer().fit(question_list + [question])

    # Vectorize the question list and the input question
    question_list_vec = vectorizer.transform(question_list)
    question_vec = vectorizer.transform([question])

    # Calculate cosine similarities
    similarities = cosine_similarity(question_vec, question_list_vec)

    # If the maximum similarity is greater than the threshold, return True
    if similarities.max() > similarity_threshold:
        return True

    return False


question_list = [
    "What do you think we're going to learn?",
    "Are there any other important things we need to know about this lesson?",
    "Are there any other points we should know about this lesson?",
    "Why is that?",
    "What else is significant?",
    "What else are you studying?",
    "What else can you tell me about the document?",
    "What else can you tell me about the article?",
    "What did they find?",
    "What else was said?",
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

for i in question_list:
    if not is_general_question(i):
        print(i)

        print(is_general_question(i))
    print(is_general_question(i))