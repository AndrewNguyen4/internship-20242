# Flask-based Text Suggestion Demo
from flask import Flask, request, render_template, jsonify
import html
import pickle
import re
from tensorflow.keras.models import load_model
from FFN import FFNLanguageModel, predict_ffn_next, complete_ffn
from LSTM import predict_lstm_next, complete_lstm, max_seq_len
from ngram.ngrams_interpolate import NGramPredictor

# Load tokenizer
with open("models/word2vec_model_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load models
print("Loading models... This might take a while...")
ngram2 = NGramPredictor.load_model("models/ngram_model_2.pkl")
ffn_model = load_model("models/ffn_nextword_model.keras", custom_objects={"FFNLanguageModel": FFNLanguageModel})
lstm_model = load_model("models/LSTM-pretrained_embs-not_frozen/epoch_5.keras")
print("Models loaded.")

app = Flask(__name__)

def tokenize(text):
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("user_input", "")
    model_name = data.get("model", "")

    tokens = tokenize(user_input)
    ends_with_space = user_input.endswith(" ")
    partial = "" if ends_with_space or not tokens else tokens[-1]
    context = tokens if ends_with_space else tokens[:-1]

    suggestions = []

    if model_name == "2-gram":
        suggestions = ngram2.complete_word(context, partial, k=3) if partial else ngram2.predict_next_word(context, k=3)
    elif model_name == "FFN":
        suggestions = complete_ffn(context, partial, k=3) if partial else predict_ffn_next(context, k=3)
    elif model_name == "LSTM":
        suggestions = complete_lstm(context, partial, k=3) if partial else predict_lstm_next(context, k=3)

    if suggestions is None: 
        suggestions = []
    
    buttons = [
        f'<button class="suggestion-btn" onclick="useSuggestion({repr(w)})">{html.escape(w)}</button>'
        for w in suggestions
    ]
    return "".join(buttons)  



if __name__ == "__main__":
    app.run(debug=False)
