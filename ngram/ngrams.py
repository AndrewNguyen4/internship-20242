import re
import os
import pickle  # For saving and loading
from collections import defaultdict, Counter
from nltk.util import ngrams


def tokenize(text):
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())

class NGramPredictor:
    def __init__(self, n=3):
        self.n = n  # Define the n-gram size
        self.ngram_counts = defaultdict(Counter)  # Store n-gram counts
    
    def train(self, training_file):
        """Train the n-gram model using a text file."""
        count = 0
        with open(training_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        for sentence in sentences:
            count += 1
            if count % 10000 == 0:
                print(count)
            if count > 2900000: break
            tokens = tokenize(sentence) 
            if len(tokens) < self.n:
                continue
            
            # Create n-grams and count occurrences
            for ngram in ngrams(tokens, self.n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"):
                prefix, next_word = tuple(ngram[:-1]), ngram[-1]
                self.ngram_counts[prefix][next_word] += 1

    def predict_next_word(self, prefix, num_of_preds=1):
        """Given a prefix, predict the most likely next word."""
        prefix = tuple(prefix[-(self.n-1):])  # Last (n-1) words
        if prefix in self.ngram_counts:
            predictions = [word for word, _ in self.ngram_counts[prefix].most_common(num_of_preds)]
            return predictions if predictions else None  # Return list or None
        return None  # No prediction available

    def complete_word(self, prefix, partial_word, num_of_preds=1):
        """Given a prefix and partial word, complete the most likely full word."""
        prefix = tuple(prefix[-(self.n-1):]) 
        possible_words = [word for word in self.ngram_counts[prefix] if word.startswith(partial_word)]
        
        if possible_words:
            sorted_words = sorted(possible_words, key=lambda w: self.ngram_counts[prefix][w], reverse=True)
            predictions = sorted_words[:num_of_preds]
            return predictions if predictions else None  # Return list or None
        return None  # No suitable completion found

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename):
        with open(filename, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model

print("Start")


if __name__ == "__main__":
    model = NGramPredictor(n=3)  

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_PATH = os.path.join(BASE_DIR, r"../mobiletext/sent_train_clean.txt")

    # model.train(TRAIN_PATH)
    # model.save_model("ngram_model_3.pkl")

    # loaded_model = NGramPredictor.load_model("ngram_model_3.pkl")

    # prefix = ["a", "red"]
    # partial_word = "c"
    # completion = loaded_model.complete_word(prefix, partial_word)
    # print(f"Completion for '{partial_word}': {completion}")

    # prefix = ["best", "of"]
    # prediction = loaded_model.predict_next_word(prefix)
    # print(f"Predicted next word: {prediction}")

