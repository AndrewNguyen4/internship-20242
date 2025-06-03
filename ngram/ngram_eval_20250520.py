import re
import os
import random
import time
from ngrams_interpolate import NGramPredictor  
import pickle

# Note: This file differs from ngram_eval.py in that it makes evaluation more uniform with that of other models.

# Load trained N-gram model
def load_ngram_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Tokenization function
def tokenize(text):
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())

# Modified test function for full sequence evaluation
def test_ngram_topk_performance(test_file, ngram_model, top_k=3, lambdas = [0.4, 0.3, 0.3]):
    correct = 0
    total = 0

    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = tokenize(line.strip())
            if len(tokens) < 2:
                continue

            for i in range(1, len(tokens)):
                prefix = tokens[:i]
                target = tokens[i]
                predictions = ngram_model.predict_next_word(prefix, k=top_k, lambdas = lambdas)

                if predictions:
                    pred_words = [word for word, _ in predictions] if isinstance(predictions[0], tuple) else predictions
                    if target in pred_words:
                        correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    print("Evaluation of Sequential Next-Word Prediction (N-gram)")
    print(f"Top-{top_k} Accuracy: {accuracy:.2%} ({correct}/{total})")


def main(ngram_model_path, test_text_file, top_k=3):
    print("Loading N-gram model...")
    ngram_model = NGramPredictor.load_model(ngram_model_path)

    print("\nEvaluating Top-K Sequential Word Predictions:")
    start = time.time()
    test_ngram_topk_performance(test_text_file, ngram_model, top_k)
    print("Execution time:", time.time() - start, "seconds")    

# Example usage
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ngram_model_path = os.path.join(BASE_DIR, r"../models/ngram_model_3.pkl")
    test_text_file = os.path.join(BASE_DIR, r"C../mobiletext/sent_test_clean.txt")
    main(ngram_model_path, test_text_file, top_k=3)
