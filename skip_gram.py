import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pickle
import os
from tqdm import tqdm

# Step 1: Load and preprocess data (dummy data)
def load_corpus(file_path):
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip().lower() for line in f if line.strip()]

# Step 2: Tokenize and build vocabulary
def build_vocab(corpus, vocab_size=10000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(corpus)
    return tokenizer

# Step 3: Generate skip-gram pairs with negative samples
def generate_training_data(tokenizer, corpus, window_size=2, num_ns=4):
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(tokenizer.num_words)
    sequences = tokenizer.texts_to_sequences(corpus)
    targets, contexts, labels = [], [], []

    for seq in tqdm(sequences):
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            seq, vocabulary_size=tokenizer.num_words, window_size=window_size, sampling_table=sampling_table
        )
        for target, context in positive_skip_grams:
            context_class = tf.expand_dims(tf.cast([context], dtype=tf.int64), 1)
            negative_samples, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=tokenizer.num_words
            )

            # Build context and label vectors (for one target word)
            # Append each element from the training example to global lists.
            targets.extend([target] * (1 + num_ns))
            contexts.extend([context] + list(negative_samples.numpy()))
            labels.extend([1] + [0] * num_ns)

    return np.array(targets), np.array(contexts), np.array(labels)

# Step 4: Define Word2Vec model with flexibility
class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, **kwargs):
        super(Word2Vec, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.target_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim, name="w2v_target"
        )
        self.context_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim, name="w2v_context"
        )

    def call(self, inputs):
        target, context = inputs
        target_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dot_product = tf.reduce_sum(tf.multiply(target_emb, context_emb), axis=1)
        return dot_product

    def predict_top_k(self, word, tokenizer, k=3):
        idx = tokenizer.word_index.get(word, tokenizer.word_index[tokenizer.oov_token])
        emb = self.target_embedding(tf.constant([idx]))
        all_embs = self.context_embedding.embeddings
        similarities = tf.matmul(emb, all_embs, transpose_b=True)
        top_k = tf.math.top_k(similarities, k=k)
        reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
        return [(reverse_word_index.get(i.numpy(), '<UNK>'), s.numpy()) for i, s in zip(top_k.indices[0], top_k.values[0])]

    def predict_next_word(self, context_text, tokenizer, k=5):
        tokens = tokenizer.texts_to_sequences([context_text])[0]
        if not tokens:
            return []
        context_emb = tf.reduce_mean(self.target_embedding(tf.constant(tokens)), axis=0, keepdims=True)
        similarities = tf.matmul(context_emb, self.context_embedding.embeddings, transpose_b=True)
        top_k = tf.math.top_k(similarities, k=k)
        reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
        return [(reverse_word_index.get(i.numpy(), "<UNK>"), s.numpy()) for i, s in zip(top_k.indices[0], top_k.values[0])]

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Step 5: Train model
def train_model(corpus, embedding_dim=256, window_size=2, num_ns=6, epochs=8, model_save_path=r"word2vec_model.keras"):
    tokenizer = build_vocab(corpus)
    print(f"✅ Vocabulary size: {tokenizer.num_words}")


    targets, contexts, labels = generate_training_data(tokenizer, corpus, window_size, num_ns)
    print(f"✅ Training pairs: {len(targets)}")

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(buffer_size=10000).batch(1024, drop_remainder=True)

    model = Word2Vec(vocab_size=tokenizer.num_words, embedding_dim=embedding_dim)
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    print(f"Starting training")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        batches = tqdm(dataset, desc=f"Training", leave=False)
        for (x_batch, y_batch) in batches:
            targets, contexts = x_batch
            loss = model.train_on_batch((targets, contexts), y_batch)
            epoch_loss += loss
            batches.set_postfix(loss=f"{loss:.4f}")
        print(f"Epoch {epoch + 1} completed. Avg Loss: {epoch_loss / len(dataset):.4f}\n")

    # Save full model
    model.save("word2vec_model.keras")

    # Save tokenizer
    with open("word2vec_model_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    return model, tokenizer

def load_model_and_tokenizer(model_path, embedding_dim=None):
    model = tf.keras.models.load_model(model_path, custom_objects={"Word2Vec": Word2Vec})

    tokenizer_path = model_path + "_tokenizer.pkl"
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


# if __name__ == "__main__":
#     corpus = load_corpus(r"/mobiletext/sent_dev_clean.txt")
#     model, tokenizer = train_model(corpus)

#     # Test prediction
#     print("\nTop predictions for 'best':")
#     print(model.predict_top_k("best", tokenizer))

#     print("\nTop predictions for 'look':")
#     print(model.predict_top_k("look", tokenizer))

#     print("\nTop predictions for 'place':")
#     print(model.predict_top_k("place", tokenizer))

