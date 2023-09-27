import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import re
from scipy.sparse import hstack

train = pd.read_csv("C:/Users/teots/OneDrive/Υπολογιστής/Creating Queries For Batch Indexing/Final_Dataset_Finished_Processing_Train_Val_Test/train.csv").dropna(subset=['Document'])
test = pd.read_csv("C:/Users/teots/OneDrive/Υπολογιστής/Creating Queries For Batch Indexing/Final_Dataset_Finished_Processing_Train_Val_Test/test.csv").dropna(subset=['Document'])
val = pd.read_csv("C:/Users/teots/OneDrive/Υπολογιστής/Creating Queries For Batch Indexing/Final_Dataset_Finished_Processing_Train_Val_Test/val.csv").dropna(subset=['Document'])

tfidf_vectorizer = TfidfVectorizer()
print("Vectorize document...")
X_train_text = tfidf_vectorizer.fit_transform(train['Document'])
X_val_text  = tfidf_vectorizer.transform(val['Document'])
X_test_text = tfidf_vectorizer.transform(test['Document'])
print("Vectorization finished")

X_train_numeric = train[['Num_Emoji', 'Num_Bad_Words']].values
X_val_numeric = val[['Num_Emoji', 'Num_Bad_Words']].values
X_test_numeric = test[['Num_Emoji', 'Num_Bad_Words']].values

print(type(X_train_text))
print(X_train_numeric.shape)
# Combine numerical and text features
X_train = hstack((X_train_text, X_train_numeric))
X_val = hstack((X_val_text, X_val_numeric))
X_test = hstack((X_test_text, X_test_numeric))

# Target variable
y_train = train['Credibility']
y_val = val['Credibility']
y_test  = test['Credibility']

dtc = DecisionTreeClassifier(criterion='log_loss', splitter='random', random_state=42)
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