# Sentiment Analysis on IMDB Reviews Dataset

## Overview
This project focuses on sentiment analysis of the IMDB Reviews dataset for a Kaggle competition. The goal is to use Natural Language Processing (NLP) techniques to categorize movie reviews as either 'positive' or 'negative'.

## Data Description
- **Dataset**: IMDB Dataset.csv
- **Content**: Movie reviews with corresponding sentiment labels (positive/negative).
- **Visualization**: Initial exploration with `.describe()`, `.head()`, and `.info()` for understanding the dataset structure.

## Data Preprocessing
- **Normalization**: Converting reviews to lowercase and stripping white spaces.
- **Cleaning**: Removing HTML tags from the reviews.
- **Stopwords Removal**: Eliminating common English stopwords to focus on key informative words.
- **Tokenization**: Applying NLTK's `word_tokenize` to convert reviews into tokens.

## Feature Engineering
- **TF-IDF Vectorization**: Transforming text data into numerical form using TfidfVectorizer to emphasize important words and diminish the impact of frequently occurring words.

## Model Development
1. **Support Vector Machine (SVM)**:
   - Implementation of an SVM model for classification.
   - Performance evaluation using classification reports.

2. **Neural Network**:
   - Dimensionality reduction using TruncatedSVD to prepare data for neural network processing.
   - Building and training a Sequential model with Dense layers.
   - Model evaluation based on accuracy.

## Libraries Used
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TensorFlow

## How to Run
Ensure all required libraries are installed. Execute the script in a Python environment to perform the sentiment analysis. The script includes data preprocessing, model training, and evaluation.

## Results
The project successfully categorizes IMDB movie reviews into 'positive' or 'negative' sentiments, showcasing the effectiveness of NLP techniques in text classification tasks.

