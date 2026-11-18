# AI Review Analyzer

A bilingual machine learning application that analyzes customer reviews and classifies their sentiment as positive or negative.

**Languages:** Russian and English

**Live Demo:** [Open AI Review Analyzer](https://ai-review-analyzer-ksbpgbct57bslcquu3e74e.streamlit.app/)

## Project Overview

AI Review Analyzer is an educational Natural Language Processing (NLP) project built with Python and Scikit-learn.

The application accepts a customer review and predicts whether its sentiment is positive or negative.

The project uses:
- TF-IDF for text feature extraction.
- Logistic Regression for sentiment classification.
- Scikit-learn Pipeline to combine text vectorization and classification.
- Streamlit to provide a web interface.

The model is trained on examples in both Russian and English.

## Features

- Sentiment classification: Positive or Negative.
- Russian and English review support.
- Prediction confidence display.
- Interactive web interface.
- Machine learning model training and evaluation.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- Pickle

## Project Structure

```text
ai-review-analyzer/
│
├── app.py                 # Streamlit web application
├── train_model.py         # Model training and evaluation
├── sentiment_analyzer.py  # Initial rule-based analyzer
├── sentiment_model.pkl    # Trained machine learning model
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## How It Works

1. The user enters a customer review.
2. TF-IDF converts the text into numerical features.
3. Logistic Regression predicts the sentiment.
4. The application displays the predicted sentiment and model confidence.

## Dataset

The current training dataset contains 48 manually written review examples:

- 12 positive Russian reviews.
- 12 negative Russian reviews.
- 12 positive English reviews.
- 12 negative English reviews.

The dataset is divided into training and testing subsets using an 75/25 split.

## Installation

Clone the repository:

```bash
git clone https://github.com/Avva-Dorofey/ai-review-analyzer.git
```

Navigate to the project directory:

```bash
cd ai-review-analyzer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

## Train the Model

To train the model again, run:

```bash
python train_model.py
```

The training script evaluates the model and saves the trained pipeline to `sentiment_model.pkl`.

## Current Limitations

- The training dataset is very small.
- Model performance has not been validated on a large independent dataset.
- Predictions may be inaccurate, especially for complex or mixed-sentiment reviews.
- The model supports binary classification only: positive and negative.
- The displayed confidence should not be interpreted as a guarantee of prediction correctness.

This project is an educational prototype and is not intended for production use.

## Author

Dorofey
