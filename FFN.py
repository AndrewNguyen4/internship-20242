import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.saving import register_keras_serializable
from skip_gram import Word2Vec
from tqdm import tqdm

# Load tokenizer
TOKENIZER_PATH = "models/word2vec_model_tokenizer.pkl"
MODEL_CHECKPOINT = "models/ffn_nextword_model.keras"
WINDOW_SIZE = 3

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = tokenizer.num_words
embedding_dim = 256 

# Load pretrained Word2Vec
w2v_model = Word2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim)
w2v_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
w2v_model = load_model("models/word2vec_model_finetuned.keras",
                      custom_objects={"Word2Vec": Word2Vec},
                      compile=False)
pretrained_embeddings = w2v_model.get_layer("w2v_target").get_weights()[0]

@register_keras_serializable()
class FFNLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, window_size, pretrained_weights, **kwargs):
        super().__init__(**kwargs)
        # store them
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size   = window_size
        self.pretrained_weights = pretrained_weights
        # build layers exactly as before
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[pretrained_weights],
            trainable=False,
            name="ffn_embedding",
        )
        self.flatten      = tf.keras.layers.Flatten(name="ffn_flatten")
        self.dense1       = tf.keras.layers.Dense(128, activation="relu", name="ffn_dense1")
        self.output_layer = tf.keras.layers.Dense(vocab_size, name="ffn_output")

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.output_layer(x)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        # ignore config contents, re-use globals
        from FFN import vocab_size, embedding_dim, WINDOW_SIZE, pretrained_embeddings
        return cls(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            window_size=WINDOW_SIZE,
            pretrained_weights=pretrained_embeddings
        )


    def predict_next_word(self, prefix_words, k=5):
        # prefix_words: list of tokens (strings)
        seq = tokenizer.texts_to_sequences([" ".join(prefix_words)])[0]
        seq = seq[-self.window_size:]
        seq = [0] * (self.window_size - len(seq)) + seq
        logits = self.predict(np.array([seq]), verbose=0)[0]
        
        idx2word = {v: k for k, v in tokenizer.word_index.items()}
        sorted_ids = logits.argsort()[::-1]
        
        # Filter out <UNK> (typically index 0)
        predictions = []
        for i in sorted_ids:
            if i == 0:  # skip <UNK>
                continue
            word = idx2word.get(i)
            if word:
                predictions.append(word)
            if len(predictions) == k:
                break
        return predictions

    def complete_word(self, prefix_words, partial, k=5):
        candidates = self.predict_next_word(prefix_words, k=100)  
        filtered = [w for w in candidates if w.startswith(partial)]
        return filtered[:k] if filtered else []


# Initialize or load FFN model
if os.path.exists(MODEL_CHECKPOINT):
    model = load_model(MODEL_CHECKPOINT, custom_objects={"FFNLanguageModel": FFNLanguageModel}, compile=False)
else:
    model = FFNLanguageModel(vocab_size, embedding_dim, WINDOW_SIZE, pretrained_embeddings)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )

# Training and evaluation 
def generate_ffn_training_data(corpus, tokenizer, window_size=WINDOW_SIZE):
    sequences = tokenizer.texts_to_sequences(corpus)
    x, y = [], []
    for seq in sequences:
        for i in range(window_size, len(seq)):
            x.append(seq[i-window_size:i])
            y.append(seq[i])
    return np.array(x), np.array(y)


def test_ffn_model(test_file, model, tokenizer, window_size=WINDOW_SIZE, top_k=3):
    total = correct = 0
    # idx2word = {v: k for k, v in tokenizer.word_index.items()}
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in tqdm(lines, desc="Evaluating FFN"):
        tokens = tokenizer.texts_to_sequences([line])[0]
        if len(tokens) <= window_size:
            continue
        for i in range(window_size, len(tokens)):
            context = tokens[i-window_size:i]
            target = tokens[i]
            logits = model.predict(np.array([context]), verbose=0)[0]
            top_ids = logits.argsort()[-top_k:][::-1]
            if target in top_ids:
                correct += 1
            total += 1
    acc = correct/total if total else 0
    print(f"FFN Top-{top_k} Accuracy: {acc:.2%} ({correct}/{total})")

# --- Expose API for demo.py ---
__all__ = ["tokenizer", "model", "FFNLanguageModel", "predict_ffn_next", "complete_ffn"]

def predict_ffn_next(prefix_words, k=5):
    return model.predict_next_word(prefix_words, k)

def complete_ffn(prefix_words, partial, k=5):
    return model.complete_word(prefix_words, partial, k)
