import re
import string
import nltk
from nltk.corpus import stopwords
from glove import Corpus, Glove
from pprint import pprint
# nltk.download('stopwords')
# nltk.download('punkt') 

stops = stopwords.words('english')

def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """
    Function that cleans the input text by going to:
    - remove links
    - remove special characters
    - remove numbers 
    - remove stopwords
    - convert to lowercase
    - remove excessive white spaces
    Arguments:
        text (str): text to clean
        remove_stopwords (bool): whether to remove stopwords
    Returns:
        str: cleaned text
    """
    # remove links
    text = re.sub(r'http\S+', "", text)

    # remove \n
    text = text.replace(r"\n", " ")

    # remove \n
    text = text.replace(r"\t", " ")

    # remove numbers and special characters
    text = re.sub("[^A-Za-z]+", " ", text)

    # remove stopwords
    if remove_stopwords:
        # 1. create tokens
        tokens = nltk.word_tokenize(text)
        #2. check if it is a stopword
        tokens = [w.lower().strip() for w in tokens if w.lower() not in stops]
        return tokens
    
import pandas as pd

df = pd.read_csv("Final_Dataset_Misinformation.csv")

# pprint(preprocess_text(df.Document[0], True))



# Apply preprocess function to the documents with pandas
df["cleaned"] = df.Document.apply(
    lambda x: preprocess_text(str(x), remove_stopwords=True)
)

# I am transforming the cleaned to list in order to use them to train the word2vec
texts = df.cleaned.tolist()

#Creating a corpus object
corpus = Corpus() 

#Training the corpus to generate the co-occurrence matrix which is used in GloVe
corpus.fit(texts, window=10)

glove = Glove(no_components=5, learning_rate=0.05) 
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove.model')