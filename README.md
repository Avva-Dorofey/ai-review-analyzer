
# AI Review Analyzer

A Machine Learning project for sentiment analysis of customer reviews.

## Project Overview

AI Review Analyzer is a beginner-friendly Natural Language Processing (NLP) project that classifies customer reviews into positive and negative categories.

The project uses TF-IDF for text feature extraction and Logistic Regression for sentiment classification.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Natural Language Processing (NLP)

## Project Structure

```text
ai-review-analyzer/
│
├── sentiment_analyzer.py
├── train_model.py
├── requirements.txt
└── README.md
```

## How It Works

1. Collects customer review text.
2. Preprocesses text data.
3. Converts text into numerical features using TF-IDF.
4. Trains a Logistic Regression classifier.
5. Evaluates model performance on test data.
6. Saves the trained model.

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Model Training

Run:

```bash
python train_model.py
```

## Current Limitations

- The initial training dataset is small.
- Model performance requires further evaluation.
- The project is an educational prototype.

## Author

Dorofey
