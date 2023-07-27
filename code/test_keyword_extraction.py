import re
from rake_nltk import Rake

def extract_keywords(sentences, num_keywords):
    r = Rake()  # Uses stopwords for English from NLTK, and all punctuation characters.
    keywords = []
    
    for sentence in sentences:
        # Remove numbers
        sentence = re.sub(r'\b\d+\b', '', sentence)
        r.extract_keywords_from_text(sentence)
        extracted_keywords = r.get_ranked_phrases()[:num_keywords]  # To get keyword phrases ranked highest to lowest.
        keywords.extend(extracted_keywords)
    
    return keywords

sentences = ["U.S. government agencies, such as the National Aeronautics and Space Administration (NASA) and National Oceanic and Atmospheric Administration, have identified many challenges in which sustainability can make a positive contribution.", "These include climate change, decreasing supplies of clean water, loss of ecological systems, degradation of the oceans, air pollution, an increase in the use and disposal of toxic substances, and the plight of endangered species.", "41 Progress toward solving these challenges depends in part on deciding who should help pay for the protection of global environmental resources; this is an issue of both environmental and distributive justice."]

keywords = extract_keywords(sentences, 2)
print(f"Keywords: {keywords}")
print(len(keywords))
