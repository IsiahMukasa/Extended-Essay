# Import
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.utils import resample


# Load datasets and do light pre-processing
training_data = pd.read_parquet("../data/training.parquet")[["text", "hate_speech_score"]]
training_data["hate"] = ["HATE" if label > 0.3 else "NOHATE" for label in training_data["hate_speech_score"]]
training_data = resample(training_data, stratify=training_data['hate'], n_samples=50000, random_state=42)

testing_data = pd.read_csv("../data/testing.csv")


# Preprocess text
def preprocess_text(text):
    # Remove special characters and links
    text = re.sub(r'http\S+|www\S+|https\S+|@\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply proper preprocessing to training text and get each sentence as a list of words
training_sentences = [sentence.lower().split() for sentence in training_data["text"].apply(preprocess_text)]
testing_sentences = [sentence.lower().split() for sentence in testing_data["text"].apply(preprocess_text)]
sentences = training_sentences + testing_sentences

# Train Word2Vec model with only training data
model = Word2Vec(sentences, window=5, min_count=1, workers=4)


# Function to create a list of vectors from a sentence
def get_sentence_vector(sentence):
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


# Convert text to sequences
training_vectors = np.array([get_sentence_vector(sentence) for sentence in training_sentences])
testing_vectors = np.array([get_sentence_vector(sentence) for sentence in testing_sentences])

# Create train and test sets
X_train = training_vectors
Y_train = training_data["hate"]
X_test = testing_vectors
Y_test = testing_data["hate"]

# Create an SVM classifier
clf = svm.SVC()

# Train the classifier
clf.fit(X_train, Y_train)

# Predict the labels for the testing dataset
Y_pred = clf.predict(X_test)

# Print report of Precision, Recall and F1-Score
report = classification_report(Y_test, Y_pred)
print(f"Classification Report:\n{report}")