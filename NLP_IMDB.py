# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:03:35 2023

@author: Fayssal
"""

""" IMBD Review - Sentiment Analysis - Kaggle Competition """

# Importing the necessary libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

# Import the data from IMDB Dataset
df = pd.read_csv('C://Users/Fayssal/Desktop/Neoma MSc TOUS LES COURS/Kaggle & Practice/Movies/IMDB Dataset.csv')

# Visualize the data
df.describe()
df.head(10)
df.info()

"""Data cleaning and pre-processing"""

# Put all the reviews to lower case and remove HTML tags

df['review'] = df['review'].str.lower()
df['review'] = df['review'].str.strip()

# HTML tags
df['clean_review'] = df['review'].str.replace(r'<[^<>]*>', '', regex=True)

# Remove stopwords, we need to gather the key informations
nltk.download('stopwords')

# Store stopwords and remove them from clean review column
stop = stopwords.words('english')
df['clean_review'] = df['clean_review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# We tokenize the results, and we go for the modelisation
from nltk.tokenize import word_tokenize

# Creating a function which will be applied for our column
def tokenize_text(text):
    return word_tokenize(text.lower())

df['clean_review'] = df['clean_review'].apply(tokenize_text)

# Reconvert token into strings
df['clean_review'] = df['clean_review'].apply(lambda x: ' '.join(x))

""" Train the data with NLP """

from sklearn.model_selection import train_test_split
# Define the features (the reviews after cleaning) & the target (sentiment)
X = df['clean_review']
df_dummies = pd.get_dummies(df['sentiment'])
y = df_dummies['positive']

# Train & Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizer TF-IDF - We use this method to reduce the weight of the words that are too much present
# And to take into account the importance of the other words
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

# Adapt the vectorizer to the train dataset and transform it
X_train_tfidf = vectorizer.fit_transform(X_train)

# Same for the test dataset
X_test_tfidf = vectorizer.transform(X_test)

"SVM Model"

from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Create and train SVM model
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)

# Prediction and analyze of SVM's performance
y_pred_svm = svm_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred_svm))

"Neural Network Model"

from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

svd = TruncatedSVD(n_components=300)  # We reduce the components of the matrix, to make it lighter 
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=300))  # Dimension = 300 as it is the number of components of our svd
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_svd, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_svd, y_test)
print(f'Accuracy: {accuracy}')




