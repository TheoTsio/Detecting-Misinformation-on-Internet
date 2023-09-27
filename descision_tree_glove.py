import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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


train = pd.read_csv("C:/Users/teots/OneDrive/Υπολογιστής/Creating Queries For Batch Indexing/Final_Dataset_Finished_Processing_Train_Val_Test/train.csv").dropna(subset=['Document'])
test = pd.read_csv("C:/Users/teots/OneDrive/Υπολογιστής/Creating Queries For Batch Indexing/Final_Dataset_Finished_Processing_Train_Val_Test/test.csv").dropna(subset=['Document'])
val = pd.read_csv("C:/Users/teots/OneDrive/Υπολογιστής/Creating Queries For Batch Indexing/Final_Dataset_Finished_Processing_Train_Val_Test/val.csv").dropna(subset=['Document'])

word2vec_vectorized = Word2Vec.load("word2vec_my_trained_models\healthword2vec_window_10_cbow.model")

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    with open(gloveFile, encoding='utf8') as f:
        content = f.readlines()
    model = {}
    for line in content:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    print('Done.', len(model), ' words loaded!')
    return model

# glove_model = loadGloveModel('glove.6B.50d.txt')
# glove_model = loadGloveModel('glove.6B.100d.txt')
# glove_model = loadGloveModel('glove.6B.200d.txt')
# glove_model = loadGloveModel('glove.6B.300d.txt')
glove_model = loadGloveModel('glove.42B.300d.txt')


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

    vector = [glove_model[word] for word in tokenized_document if word in glove_model.keys()]
    doc_vector = np.mean(vector, axis=0)

    return doc_vector

print(val.shape)

# Apply preprocess function to the documents with pandas
print("Vectorization Starting for train...")
train['Document'] = train.Document.apply(
    lambda x: vectorize_document(str(x))
)
print("Vectorization Starting for val...")
val['Document'] = test.Document.apply(
    lambda x: vectorize_document(str(x))
)
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
X_train = hstack((csr_matrix(X_train_text.tolist()), X_train_numeric))
X_val = hstack((csr_matrix(X_val_text.tolist()), X_val_numeric))
X_test = hstack((csr_matrix(X_test_text.tolist()), X_test_numeric))

# Target variable
y_train = train['Credibility']
y_val = val['Credibility']
y_test  = test['Credibility']

dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)

# Predictions on test set
y_val_pred = dtc.predict(X_val)
y_test_pred = dtc.predict(X_test)

# Evaluate the val
accuracy = accuracy_score(y_val, y_val_pred)
print("Val Accuracy:", accuracy)
f1 = f1_score(y_val, y_val_pred, average=None)
f1_micro = f1_score(y_val, y_val_pred, average='micro')
f1_macro = f1_score(y_val, y_val_pred, average='macro')
f1_weighted = f1_score(y_val, y_val_pred, average='weighted')
print("Val F1", f1)
print("Val F1 micro", f1_micro)
print("Val F1 macro", f1_macro)
print("Val F1 weighted", f1_weighted, end='\n\n')


# Evaluate the test
accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", accuracy)
f1 = f1_score(y_test, y_test_pred, average=None)
f1_micro = f1_score(y_test, y_test_pred, average='micro')
f1_macro = f1_score(y_test, y_test_pred, average='macro')
f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
print("Test F1", f1)
print("Test F1 micro", f1_micro)
print("Test F1 macro", f1_macro)
print("Test F1 weighted", f1_weighted)