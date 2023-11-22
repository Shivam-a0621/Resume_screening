import re
from nltk.stem import PorterStemmer

class TextCleaner:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def clean(self, text):
        # Your cleaning logic
        cleantxt = re.sub("https\S+", "", text)
        cleantxt = re.sub(r'[^a-zA-Z0-9\s]', '', cleantxt)
        cleantxt = cleantxt.lower()
        cleantxt = re.sub('\r\n', '', cleantxt)
        cleantxt = re.sub('\n', '', cleantxt)
        cleantxt = re.sub('\r', '', cleantxt)
        words = cleantxt.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        cleantxt = ' '.join(stemmed_words)
        
        return cleantxt

# Example usage:
text_cleaner = TextCleaner()
cleaned_text = text_cleaner.clean("Your input text here")
print(cleaned_text)
