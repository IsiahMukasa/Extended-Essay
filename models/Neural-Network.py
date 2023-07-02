import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

# Load datasets
training = pd.read_parquet("../data/training.parquet")[["text","hate_speech_score"]]
training["hate"] = ['HATE' if label > 0.4 else 'NOHATE' for label in training["hate_speech_score"]]

testing = pd.read_csv("../data/testing.csv")


# Extract the text column from the DataFrame
train_data = training['text'].tolist()
test_data = testing['Comments'].tolist()

# Preprocess text
def preprocess_text(text):
    # Remove special characters and links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing to training text
training_sentences = training["text"].apply(preprocess_text)
testing_sentences = testing["Comments"].apply(preprocess_text)

# Train Word2Vec model
model = Word2Vec(training_sentences, window=5, min_count=1, workers=4)


# Function to convert a sentence to a feature vector using word2vec
def get_sentence_vector(sentence):
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


# Create an embedding matrix
embedding_matrix = model.wv.vectors

# Get the vocabulary size and embedding dimension
vocab_size = len(model.wv.key_to_index)
embedding_dim = model.vector_size

# Convert text to sequences of Word2Vec embeddings
sequences = []
for text in training["text"]:
    seq = []
    for word in text.split():
        if word in model.wv:
            seq.append(model.wv[word])
    sequences.append(seq)

# Pad sequences
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post")


# Create train and test sets
X_train = np.vstack([get_sentence_vector(sentence.split()) for sentence in training_sentences])
Y_train = training["hate"].apply(lambda x: 1 if x == "HATE" else 0) # RNN can only classify binary classes
X_test = np.vstack([get_sentence_vector(sentence) for sentence in testing_sentences])
Y_test = testing["Hate"]


# Create the RNN model with embedding, LSTM and dense output layers
rnn = Sequential()
rnn.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
rnn.add(LSTM(64))
rnn.add(Dense(1, activation="sigmoid"))

# Compile model
rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
rnn.fit(X_train, Y_train, epochs=10, batch_size=32)

prediction = rnn.predict(X_test)

# Predict the labels for the testing dataset
Y_test = ["HATE" if i == 1 else "NOHATE" for i in Y_test]
Y_pred = ["HATE" if i > 0.5 else "NOHATE" for i in prediction]

report = classification_report(Y_test, Y_pred)
print(f"Classification Report:\n{report}")
