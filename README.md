# Smartphone Keyboard Suggestor

This project implements N-gram, Feedforward Neural Network (FFN), and LSTM models to predict and suggest the next word in the context of mobile phone messages. It includes a prototype user interface for real-time interaction built using Flask.

**Note:** This repository only contains necessary for a minimal prototype. As training was conducted on Kaggle and Google Colab to utilize GPU resources, training scripts of Word2Vec, FFN and LSTM can be found in the links below.
- [Word2Vec](https://colab.research.google.com/drive/1lO-mX_WCrWglw0LX1S1QZjh97Ok4hjBc?usp=sharing)
- [FFN](https://colab.research.google.com/drive/1SvY10YrdAKN4nOJfAM4eMYHaIlb2yMro?usp=sharing)
- [LSTM](https://bit.ly/3Hkn0Z0)  
The trained models can be found [here](https://husteduvn-my.sharepoint.com/:f:/g/personal/duc_nm225437_sis_hust_edu_vn/Ep3UlaSY2BVKtzYl_XyDPNQBnSPyKRHjmmPJy8krgtV56Q?e=XE2q72).
The original dataset can be found [here](https://digitalcommons.mtu.edu/mobiletext/).

## Features

- **N-gram Models**: Statistical models that predict the likelihood of a word given the previous words.
- **FFN Model**: A simple neural network using pretrained Word2Vec embeddings for context-based prediction.
- **LSTM Model**: A recurrent neural network capable of learning longer context dependencies.
- **Interactive Flask Web Interface**: Type messages and receive real-time next-word suggestions from any of the models.


## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/AndrewNguyen4/internship-20242.git
    cd internship-20242
    ```

2. **Create a virtual environment (recommended)**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download model files**:
    Download the file `models` from this [link](https://husteduvn-my.sharepoint.com/:f:/g/personal/duc_nm225437_sis_hust_edu_vn/Ep3UlaSY2BVKtzYl_XyDPNQBnSPyKRHjmmPJy8krgtV56Q?e=XE2q72) and unzip if necessary, then place the unzipped `models` file to the `internship-20242` directory.
    Refer to the expected folder structure below.
    

## Usage
1. Start the Flask application:
    ```sh
    python app.py
    ```

2. Open your browser and visit http://127.0.0.1:5000.
    
3. Use the input box to type a sentence. The system:

- Suggests next words when a space is typed.

- Suggests completions for partially typed words.

- Allows switching between N-gram, FFN, and LSTM models.

4. You can click on suggestions to insert them into the input box.
	   
## Folder Structure

```
internship-20242/
│
├── requirements.txt                  # Listing dependencies
├── app.py                            # Flask app logic
├── FFN.py                            # FFN model and prediction logic for app
├── LSTM.py                           # LSTM model and prediction logic for app
├── skip_gram.py                      # SkipGram model
├── models                            # Contain model data
├── mobiletext/                       # Data for training and testing (not uploaded due to size constraints)
├── ngram/                            # N-gram predictor implementation
└── templates/
    └── index.html                    # Web interface

```