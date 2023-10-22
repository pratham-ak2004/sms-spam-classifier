import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

def textPreprocessor(text):

    stemmer = PorterStemmer()
    # convert to lower case
    text = text.lower()
    # tokenize the text to individual words
    text = nltk.word_tokenize(text)
    # remove special characters and convert
    lst = []
    for word in text:
        # stopwords : which helps in formation of sentences and has no special meaning
        if word.isalnum() and word not in stopwords.words("english") and word not in string.punctuation:
            lst.append(stemmer.stem(word))

    text = " ".join(lst)

    return text
