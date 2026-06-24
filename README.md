# Twitter Sentiment Analysis with DistilBERT

## Project Overview

Social media platforms generate millions of user opinions every day, making sentiment analysis a valuable tool for businesses, marketers, and researchers seeking to understand public perception at scale.

This project develops a high-performance sentiment classification model using **DistilBERT**, a Transformer-based Natural Language Processing (NLP) model, to automatically classify Twitter posts into four sentiment categories:

* Negative
* Neutral
* Positive
* Irrelevant

The solution demonstrates how modern Transformer architectures significantly outperform traditional machine learning approaches in understanding context, language nuances, and user intent.

---

## Business Problem

Organizations rely on customer feedback from social media to:

* Monitor brand reputation
* Measure customer satisfaction
* Detect emerging issues and trends
* Analyze public reaction to products, services, and events

Manual review of thousands of tweets is impractical. An automated sentiment classification system enables real-time analysis and scalable decision-making.

---

## Dataset

The project uses a Twitter sentiment dataset containing approximately **74,000+ tweets** labeled across four sentiment classes.

### Features

* Tweet Text

### Target Variable

* Sentiment

  * Negative
  * Neutral
  * Positive
  * Irrelevant

### Data Preparation

* Removed missing tweet records
* Converted categorical sentiment labels into numerical classes
* Performed train/test split with stratification to preserve class distribution
* Tokenized text using the DistilBERT tokenizer

---

## Approach

### 1. Data Preprocessing

* Data cleaning
* Missing value handling
* Label encoding
* Train/Test splitting

### 2. Transformer-Based NLP Pipeline

Implemented a fine-tuned **DistilBERT** model using the Hugging Face Transformers library.

Key components:

* DistilBERT Tokenizer
* DistilBERT Sequence Classification Model
* Custom PyTorch Dataset
* Hugging Face Trainer API

### 3. Model Training

The model was fine-tuned on the Twitter dataset for sentiment classification across four classes.

Training included:

* Tokenization and attention masks
* Batch processing
* Validation monitoring
* Performance evaluation after each epoch

### 4. Model Evaluation

Performance was measured using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

---

## Results

| Metric    | Score      |
| --------- | ---------- |
| Accuracy  | **90.57%** |
| Precision | **90.57%** |
| Recall    | **90.57%** |
| F1-Score  | **90.56%** |

### Key Findings

* The DistilBERT model achieved strong and consistent performance across all sentiment classes.
* Negative sentiment achieved the highest classification accuracy.
* The model demonstrated strong contextual understanding of tweet language.
* Most predictions fell along the confusion matrix diagonal, indicating high classification accuracy.
* Minor confusion occurred between Neutral and Positive sentiments, which is common in real-world sentiment analysis tasks.

---

## Example Predictions

| Tweet                                              | Prediction |
| -------------------------------------------------- | ---------- |
| "I absolutely love this game. Best purchase ever!" | Positive   |
| "This update ruined everything."                   | Negative   |
| "Nvidia announced a new graphics card."            | Neutral    |
| "I want my money back."                            | Negative   |

---

## Technologies Used

* Python
* Pandas
* NumPy
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* Matplotlib
* Seaborn
* Google Colab

---

## Project Structure

```text
twitter-sentiment-analysis/
│
├── twitter_sentiment_analysis.ipynb
├── README.md
│
├── saved_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_args.bin
│
└── results/
    ├── confusion_matrix.png
    └── classification_report.png
```

---

## Impact

This project demonstrates the practical application of modern Transformer-based NLP models for large-scale sentiment analysis.

By leveraging DistilBERT, the solution achieves over **90% classification accuracy**, making it suitable for:

* Brand monitoring
* Customer experience analytics
* Social media intelligence
* Market research
* Public opinion tracking

The project highlights proficiency in:

* Natural Language Processing (NLP)
* Transformer Models
* Deep Learning
* PyTorch
* Hugging Face Ecosystem
* Model Evaluation and Deployment

---

## Future Improvements

* Hyperparameter optimization
* Cross-validation experiments
* Model deployment with Streamlit
* Real-time Twitter sentiment dashboard
* Comparison against traditional ML models (Logistic Regression, SVM, Naive Bayes)
* Fine-tuning larger Transformer models such as BERT and RoBERTa

---

## Author

**Franck Fossi**
