import re
import os
import random
from ngrams_interpolate import NGramPredictor  # Import the class

# Load trained N-gram model (dictionary of n-gram probabilities)
def load_ngram_model(model_path):
    """Loads a trained n-gram model from a file."""
    import pickle
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Tokenization function
def tokenize(text):
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())

# Test word completion
def test_word_completion(test_file, ngram_model, num_of_preds=1):
    """Evaluates word completion accuracy using an N-gram model."""
    correct = 0
    total = 0
    none = 0
    
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            sentence, answer = line.strip().split(" | ")
            prefix, partial_word = " ".join(sentence.split()[:-1]), sentence.split()[-1]

            predicted = ngram_model.complete_word(prefix, partial_word, num_of_preds)

            if not predicted:
                none += 1
            elif (partial_word + answer) in predicted:
                correct += 1

            total += 1

    accuracy = correct / total if total > 0 else 0
    none_rate = none / total if total > 0 else 0
    print("Test Word Completion")
    print(none, correct, total)
    print(f"Accuracy: {accuracy:.2%} (Top-{num_of_preds})")
    print(f"None rate: {none_rate:.2%} (Top-{num_of_preds})")

# Test word prediction
def test_word_prediction(test_file, ngram_model, num_of_preds=1):
    """Evaluates word prediction accuracy using an N-gram model."""
    correct = 0
    total = 0
    none = 0
    
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            prefix, answer = line.strip().split(" | ")
            predicted = ngram_model.predict_next_word(prefix, num_of_preds)

            if not predicted:
                none += 1
            elif answer in predicted:
                correct += 1

            total += 1

    accuracy = correct / total if total > 0 else 0
    none_rate = none / total if total > 0 else 0
    print("Test Word Prediction")
    print(none, correct, total)
    print(f"Word Prediction Accuracy: {accuracy:.2%} (Top-{num_of_preds})")
    print(f"None rate: {none_rate:.2%} (Top-{num_of_preds})")
    
# Main function to run tests
def main(ngram_model_path, test_completion_file, test_prediction_file, num_of_preds=1):
    """Loads model, runs tests, and prints results."""
    print("Loading N-gram model...")
    ngram_model = NGramPredictor.load_model(ngram_model_path)
    
    print("\nTesting Word Completion:")
    test_word_completion(test_completion_file, ngram_model, num_of_preds)

    print("\nTesting Word Prediction:")
    test_word_prediction(test_prediction_file, ngram_model, num_of_preds)
    
# Example usage
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ngram_model_path = os.path.join(BASE_DIR, r"../models/ngram_model_3.pkl" )
    test_completion_file = os.path.join(BASE_DIR, r"../mobiletext/sent_test_completion.txt")  # Word completion set
    test_prediction_file = os.path.join(BASE_DIR, r"../mobiletext/sent_test_prediction.txt")  # Word prediction set
    num_of_preds = 3 

    main(ngram_model_path, test_completion_file, test_prediction_file, num_of_preds)
