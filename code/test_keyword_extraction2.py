import re
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def generate_topic(passage, num_topics):
    stop_words = set(stopwords.words('english'))
    
    # Remove LaTeX specifics
    passage = re.sub(r'\\\(.+?\\\)', '', passage)  # remove LaTeX enclosed with \(...\)
    passage = re.sub(r'\\\{.+?\\\}', '', passage)  # remove LaTeX enclosed with \{...\}
    
    # Remove remaining special characters except for alphanumeric and space
    passage = re.sub(r'[^a-zA-Z0-9\s]', '', passage)
    
    # Tokenize and remove stop words
    texts = [word for word in word_tokenize(passage.lower()) if word not in stop_words]

    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary([texts])

    # Convert document into the bag-of-words (BoW) format = list of (token_id, token_count) tuples.
    corpus = [dictionary.doc2bow(texts)]

    # Train the model
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Get the topics
    topics = lda.print_topics(num_words=5)
    
    # Extract only the words without their weights for each topic and store them in a list
    topics_keywords = [[word_num.split('*')[1].replace('"', '') for word_num in topic[1].split(' + ')] for topic in topics]
    # Extract only the words without their weights for the first topic and store them in a list
    first_topic_keywords = [word_num.split('*')[1].replace('"', '') for word_num in topics[0][1].split(' + ')]
    
    # return topics_keywords
    return first_topic_keywords

passage = """U.S. government agencies, such as the National Aeronautics and Space Administration (NASA) and National Oceanic and Atmospheric Administration, have identified many challenges in which sustainability can make a positive contribution. These include climate change, decreasing supplies of clean water, loss of ecological systems, degradation of the oceans, air pollution, an increase in the use and disposal of toxic substances, and the plight of endangered species. 41 Progress toward solving these challenges depends in part on deciding who should help pay for the protection of global environmental resources; this is an issue of both environmental and distributive justice."""
# topics_keywords = generate_topic(passage, 3)
# for i, topic_keywords in enumerate(topics_keywords):
#     print(f"Topic {i+1}: {topic_keywords}")
topics_keywords = generate_topic(passage, 3)
print(f"Keywords: {topics_keywords}")