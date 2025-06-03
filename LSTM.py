import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re

# Load tokenizer
with open("models/word2vec_model_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = tokenizer.num_words
embedding_dim = 256
max_seq_len = 8

# Utility: tokenize
def tokenize(text):
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())

# Utility: prepare padded sequences from user input
def prepare_input_sequence(prefix_words):
    token_ids = tokenizer.texts_to_sequences([" ".join(prefix_words)])[0]
    token_ids = token_ids[-(max_seq_len - 1):]
    padded = pad_sequences([token_ids], maxlen=max_seq_len - 1, padding='pre')
    return padded

def build_lstm_model(vocab_size, embedding_dim, max_seq_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_len - 1),
        LSTM(128),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


lstm_model_path = "models/LSTM-pretrained_embs-not_frozen/epoch_5.keras"
if os.path.exists(lstm_model_path):
    model = load_model(lstm_model_path)
else:
    model = build_lstm_model(vocab_size, embedding_dim, max_seq_len)

index_to_word = {v: k for k, v in tokenizer.word_index.items()}

def predict_lstm_next(prefix_words, k=3):
    padded = prepare_input_sequence(prefix_words)
    pred_probs = model.predict(padded, verbose=0)[0]
    sorted_ids = pred_probs.argsort()[::-1]
    predictions = []
    for i in sorted_ids:
        if i == 0:  # skip <UNK>
            continue
        word = index_to_word.get(i)
        if word:
            predictions.append(word)
        if len(predictions) == k:
            break
    return predictions


def complete_lstm(prefix_words, partial_start, k=3):
    padded = prepare_input_sequence(prefix_words)
    pred_probs = model.predict(padded, verbose=0)[0]
    
    candidates = []
    for i in np.argsort(pred_probs)[::-1]:
        if i == 0:
            continue
        word = index_to_word.get(i)
        if word and word.startswith(partial_start):
            candidates.append(word)
        if len(candidates) == k:
            break
    return candidates
