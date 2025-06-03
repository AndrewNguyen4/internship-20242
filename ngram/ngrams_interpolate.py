import re
import os
import pickle  # For saving and loading
from collections import defaultdict, Counter
from nltk.util import ngrams


def tokenize(text):
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())

class NGramPredictor:
    def __init__(self, n=3):
        self.n = n
        self.ngram_counts = defaultdict(Counter) 
    
    def train(self, training_file):
        """Train the n-gram model using a text file."""
        count = 0
        with open(training_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        for sentence in sentences:
            count += 1
            if count % 10000 == 0: print(count)
            if (count >= 5000000): break
            tokens = tokenize(sentence) 
            if len(tokens) < self.n:
                continue
            
            for ngram in ngrams(tokens, self.n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"):
                prefix, next_word = tuple(ngram[:-1]), ngram[-1]
                self.ngram_counts[prefix][next_word] += 1

    def _get_weighted_word_probs(self, prefix, partial_word=None, lambdas=[0.4, 0.35, 0.25]):
        if not (sum(lambdas) == 1 and len(lambdas) == self.n):
            lambdas = [0]*(self.n - 1) + [1]

        prefix = tuple(prefix[-(self.n-1):])
        word_probs = defaultdict(float)

        for i, lambda_weight in enumerate(lambdas):
            n = self.n - i
            n_prefix = tuple(prefix[-(n-1):])

            if n_prefix in self.ngram_counts:
                total_count = sum(self.ngram_counts[n_prefix].values())

                for word, count in self.ngram_counts[n_prefix].items():
                    if word in {'<s>', '</s>'}:
                        continue
                    if partial_word is None or word.startswith(partial_word):
                        word_probs[word] += lambda_weight * (count / total_count)

        return word_probs

    def predict_next_word(self, prefix, k=1, lambdas=[0.4, 0.35, 0.25]):
        word_probs = self._get_weighted_word_probs(prefix, partial_word=None, lambdas=lambdas)
        sorted_predictions = sorted(word_probs.items(), key=lambda item: item[1], reverse=True)
        return [word for word, _ in sorted_predictions[:k]] if sorted_predictions else None

    def complete_word(self, prefix, partial_word, k=1, lambdas=[0.4, 0.35, 0.25]):
        word_probs = self._get_weighted_word_probs(prefix, partial_word=partial_word, lambdas=lambdas)
        sorted_predictions = sorted(word_probs.items(), key=lambda item: item[1], reverse=True)
        return [word for word, _ in sorted_predictions[:k]] if sorted_predictions else None
    
    def predict_next_word_with_score(self, prefix, k=1, lambdas=[0.4, 0.35, 0.25]):
        word_probs = self._get_weighted_word_probs(prefix, partial_word=None, lambdas=lambdas)
        sorted_predictions = sorted(word_probs.items(), key=lambda item: item[1], reverse=True)
        return [(word, score) for word, score in sorted_predictions[:k]] if sorted_predictions else None
    
    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename):
        with open(filename, "rb") as f:
            model = pickle.load(f)
        print(f"Ngram Model loaded.")
        return model



# if __name__ == "__main__":
#     model = NGramPredictor(n=2)  
#     # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     # model.train(os.path.join(BASE_DIR, r"../mobiletext/sent_train_clean.txt"))
#     # model.save_model("ngram_model_3.pkl")

#     loaded_model = NGramPredictor.load_model("ngram_model_3.pkl")
