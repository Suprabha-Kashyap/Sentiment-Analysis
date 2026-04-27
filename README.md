# Airline Twitter Sentiment Analysis

This project implements a Natural Language Processing (NLP) pipeline to classify the sentiment of tweets directed at major U.S. airlines. It compares different machine learning models and analyzes the key linguistic features that drive sentiment.

## 📊 Dataset
The analysis uses the **AirlineTweets.csv** dataset, which contains:
- **Text:** The raw tweet content.
- **Sentiment:** Labeled as `positive`, `negative`, or `neutral`.
- **Metadata:** Airline name, retweet count, and user location.

## 🛠 Project Workflow

### 1. Text Preprocessing
Raw tweets are cleaned using regular expressions (Regex) to:
- Remove URLs and hyperlinks.
- Remove user handles (`@mentions`).
- Remove punctuation and special characters.
- Convert all text to lowercase for uniformity.

### 2. Feature Extraction
- **TF-IDF Vectorization:** Converts text into numerical vectors, weighting words based on their importance across the dataset while filtering out common English stop words.
- **Lemmatization (Optional):** Includes a pipeline for POS-tagged lemmatization using NLTK to reduce words to their base forms.

### 3. Machine Learning Models
Two primary models were implemented and compared:
- **Linear SVC:** Explored for high-dimensional text classification (noted for potential overfitting with a 92% training vs 78% test accuracy).
- **Logistic Regression:** Provided a more robust and balanced performance (~77% test accuracy) for the multi-class problem.

### 4. Evaluation
- **Classification Report:** Detailed analysis of Precision, Recall, and F1-Score for each sentiment class.
- **Confusion Matrix:** Visualizing where the model correctly identifies sentiments and where it confuses them (e.g., neutral vs. negative).
- **Feature Importance:** Extraction of top coefficient words to identify what specifically makes a tweet "positive" or "negative."

## ⚙️ Setup & Requirements

1. **Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Dependencies:**
   ```bash
   pip install pandas scikit-learn nltk matplotlib seaborn
   ```

3. **NLTK Data:**
   The following NLTK resources are required for tokenization and lemmatization:
   ```python
   import nltk
   nltk.download(['wordnet', 'punkt', 'averaged_perceptron_tagger', 'omw-1.4'])
   ```

---
*This notebook serves as a benchmark for comparing classical ML approaches on imbalanced text datasets.*
