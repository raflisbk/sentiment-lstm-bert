# Sentiment Analysis using LSTM & BERT

## Overview

This repository contains a sentiment analysis project using deep learning models, specifically **LSTM** and  **BERT** . The goal is to classify text data into different sentiment categories (e.g., positive, negative, neutral).

## Features

* **Preprocessing Pipeline** : Text cleaning, tokenization, and vectorization using TF-IDF and word embeddings.
* **LSTM Model** : A Recurrent Neural Network (RNN) model trained for sentiment classification.
* **BERT Model** : A transformer-based model fine-tuned for sentiment analysis.
* **Handling Imbalanced Data** : SMOTE and RandomOverSampler techniques.
* **Evaluation Metrics** : Accuracy, Confusion Matrix, Classification Report.

## Repository Structure

```
📂 sentiment-lstm-bert
│── 📂 data                 # Dataset folder (if applicable)
│── 📂 models               # Saved trained models
│── 📂 utils                # Preprocessing and helper functions
│── train_lstm.py           # Training script for LSTM
│── train_bert.py           # Training script for BERT
│── evaluate.py             # Evaluation script
│── requirements.txt        # Required dependencies
│── README.md               # Project documentation
│── notebook.ipynb          # Original Jupyter Notebook
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/sentiment-lstm-bert.git
cd sentiment-lstm-bert
pip install -r requirements.txt
```

## Usage

### Training LSTM Model

```bash
python train_lstm.py
```

### Training BERT Model

```bash
python train_bert.py
```

### Evaluating the Model

```bash
python evaluate.py
```

## Dependencies

* Python 3.x
* PyTorch
* Transformers
* scikit-learn
* nltk
* imbalanced-learn
* pandas, numpy, seaborn, matplotlib

## Results & Performance

Include key performance metrics and visualizations (e.g., accuracy, loss curves, confusion matrix) here.

## Contributors

* [Mohamad Rafli Agung Subekti](https://github.com/raflisbk)

## License

This project is licensed under the MIT License.
