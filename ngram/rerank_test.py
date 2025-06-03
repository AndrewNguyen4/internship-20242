import tensorflow as tf
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import random
from ngrams_interpolate import NGramPredictor

'''
Additional Experiment: Reranking N-gram based on Word2Vec similarity
'''

class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="w2v_target")
        self.context_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="w2v_context")

    def call(self, target, context):
        target_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dot_product = tf.reduce_sum(tf.multiply(target_emb, context_emb), axis=1)
        return dot_product

def load_embeddings_and_tokenizer(model_path, tokenizer_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'Word2Vec': Word2Vec})
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Get average embedding for sentence
def get_context_embedding(sentence, tokenizer, model):
    tokens = tokenizer.texts_to_sequences([sentence])[0]
    if not tokens:
        return None
    
    # Filter OOV token
    tokens = [t for t in tokens if t < tokenizer.num_words]
    if not tokens:
        return None  # No valid tokens remain

    embeddings = model.get_layer("w2v_target")(tf.constant(tokens))
    context_embedding = tf.reduce_mean(embeddings, axis=0, keepdims=True)
    return context_embedding


# Get candidate word embedding
def get_word_embedding(word, tokenizer, model):
    idx = tokenizer.word_index.get(word)

    if idx is None or idx >= tokenizer.num_words:
        return None  
    embedding = model.get_layer("w2v_context")(tf.constant([idx]))
    return embedding


# Additional Experiment - Re-rank predictions by combining N-gram score and embedding similarity
def rerank_predictions(sentence, predictions, model, tokenizer, lambda_weight=0.5):
        
    context_emb = get_context_embedding(sentence, tokenizer, model)
    if context_emb is None:
        return predictions

    results = []
    for word, ngram_score in predictions:
        word_emb = get_word_embedding(word, tokenizer, model)
        if word_emb is None:
            sim_score = 0.0
        else:
            sim_score = cosine_similarity(context_emb.numpy(), word_emb.numpy())[0,0]
            sim_score = max(sim_score, 0.0)

        combined_score = lambda_weight * ngram_score + (1 - lambda_weight) * sim_score
        results.append((word, combined_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Load trained N-gram model (dictionary of n-gram probabilities)
def load_ngram_model(model_path):
    """Loads a trained n-gram model from a file."""
    import pickle
    with open(model_path, "rb") as f:
        return pickle.load(f)

def tokenize(text):
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())


def test_word_prediction(test_file, ngram_model, num_of_preds=6,
                          model_path=None, tokenizer_path=None, lambda_weight=0.5):

    correct = 0
    total = 0
    none = 0

    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            prefix, answer = line.strip().split(" | ")
            prefix = prefix.split()
            predictions = ngram_model.predict_next_word_with_score(prefix, k=num_of_preds)

            if predictions:
                # Re-rank predictions if model and tokenizer paths are provided
                if model_path and tokenizer_path:
                    reranked_predictions = rerank_predictions(prefix, predictions, model_path, tokenizer_path, lambda_weight)
                else:
                    reranked_predictions = predictions

                # Take only top 3 re-ranked predictions
                top_k_predictions = reranked_predictions[:3]
                predicted_words = [word for word, _ in top_k_predictions]

            else:
                predicted_words = None

            if not predicted_words:
                none += 1
            elif answer in predicted_words:
                correct += 1

            if random.random() < 0.0004:
                print(line.strip(), "\n", predictions, '\n', reranked_predictions)
            total += 1

    accuracy = correct / total if total > 0 else 0
    none_rate = none / total if total > 0 else 0
    print("Test Word Prediction")
    print(f"None predictions: {none}")
    print(f"Correct predictions: {correct}")
    print(f"Total examples: {total}")
    print(f"Word Prediction Accuracy (Top-3 after reranking): {accuracy:.2%}")
    print(f"None rate: {none_rate:.2%}")

    
def test(ngram_model_path, test_prediction_file, num_of_preds=1, model_path=None, tokenizer_path=None, lambda_weight=0.5):
    print("Loading N-gram model...")
    ngram_model = NGramPredictor.load_model(ngram_model_path)
    model, tokenizer = load_embeddings_and_tokenizer(model_path, tokenizer_path)

    print("\nTesting Word Prediction:")
    test_word_prediction(test_prediction_file, ngram_model, num_of_preds, model, tokenizer, lambda_weight)
    

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ngram_model_path = os.path.join(BASE_DIR, r"..models/ngram_model_3.pkl")  # Path to saved model
    test_prediction_file = os.path.join(BASE_DIR, r"../mobiletext/sent_test_prediction.txt")  # Word prediction test set
    model_path = os.path.join(BASE_DIR, r'../word2vec_model.keras')
    tokenizer_path = os.path.join(BASE_DIR, r'../word2vec_model_tokenizer.pkl')
    num_of_preds = 6
    lambda_weight = 0.7

    test(ngram_model_path, test_prediction_file, num_of_preds, model_path, tokenizer_path, lambda_weight)

