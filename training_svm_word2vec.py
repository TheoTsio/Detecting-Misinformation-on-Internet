import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import re
from scipy.sparse import hstack
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import nltk
from pprint import pprint
from scipy.sparse import csr_matrix
# nltk.download('stopwords')
# nltk.download('punkt') 

count = 0

train = pd.read_csv("C:/Users/teots/OneDrive/Υπολογιστής/Creating Queries For Batch Indexing/Final_Dataset_Finished_Processing_Train_Val_Test/train.csv").dropna(subset=['Document'])
test = pd.read_csv("C:/Users/teots/OneDrive/Υπολογιστής/Creating Queries For Batch Indexing/Final_Dataset_Finished_Processing_Train_Val_Test/test.csv").dropna(subset=['Document'])
val = pd.read_csv("C:/Users/teots/OneDrive/Υπολογιστής/Creating Queries For Batch Indexing/Final_Dataset_Finished_Processing_Train_Val_Test/val.csv").dropna(subset=['Document'])

word2vec_vectorized = Word2Vec.load("word2vec_my_trained_models\healthword2vec_window_10_cbow.model")

def vectorize_document(document: str) -> str:
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

    # remove \n
    document = document.replace(r"\n", " ")

    # remove \n
    document = document.replace(r"\t", " ")

    # remove numbers and special characters
    document = re.sub("[^A-Za-z]+", " ", document)
    
    #tokenizer and remove of stopwords
    stops = stopwords.words('english')
    tokenized_document = nltk.word_tokenize(document)
    tokenized_document = [w.lower().strip() for w in tokenized_document if w.lower() not in stops]

    vector = [word2vec_vectorized.wv[word] for word in tokenized_document if word in word2vec_vectorized.wv]
    doc_vector = np.mean(vector, axis=0)

    global count
    count += 1
    print("Vectorize Document", count)
    return doc_vector
print(val.shape)

# Apply preprocess function to the documents with pandas
print("Vectorization Starting for train...")
train['Document'] = train.Document.apply(
    lambda x: vectorize_document(str(x))
)
count=0
print("Vectorization Starting for val...")
val['Document'] = test.Document.apply(
    lambda x: vectorize_document(str(x))
)
count=0
print("Vectorization Starting for test...")
test['Document'] = test.Document.apply(
    lambda x: vectorize_document(str(x))
)
print('Vectorization Ended...')

X_train_text = train['Document']
X_val_text = val['Document']
X_test_text = test['Document']

X_train_numeric = train[['Num_Emoji', 'Num_Bad_Words']].values
X_val_numeric = val[['Num_Emoji', 'Num_Bad_Words']].values
X_test_numeric = test[['Num_Emoji', 'Num_Bad_Words']].values

from scipy.sparse import csr_matrix
print(len(X_val_text.tolist()), X_val_numeric.shape)
X_train_combined = hstack((csr_matrix(X_train_text.tolist()), X_train_numeric))
X_val_combined = hstack((csr_matrix(X_val_text.tolist()), X_val_numeric))
X_test_combined = hstack((csr_matrix(X_test_text.tolist()), X_test_numeric))

# Target variable
y_train = train['Credibility']
y_val = val['Credibility']
y_test  = test['Credibility']

svm_model = SVC(kernel='linear', C=10)
svm_model.fit(X_train_combined, y_train)

y_val_pred = svm_model.predict(X_val_combined)
y_test_pred = svm_model.predict(X_test_combined)

# Evaluate the val
accuracy = accuracy_score(y_val, y_val_pred)
print("Val Accuracy:", accuracy)
f1 = f1_score(y_val, y_val_pred, average=None)
print("Val F1", f1)

# Evaluate the test
accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", accuracy)
f1 = f1_score(y_test, y_test_pred, average=None)
print("Test F1", f1)