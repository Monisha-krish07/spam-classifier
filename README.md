# 📧 SMS Spam Classifier — Machine Learning Project

A complete end-to-end machine learning pipeline to classify SMS messages as **Spam** or **Ham (Not Spam)** using NLP text preprocessing and multiple classification algorithms.

---

## 🎯 Objective

Build a production-ready spam detection model using the SMS Spam Collection dataset with comprehensive text preprocessing, multiple ML classifiers, and detailed performance evaluation.

---

## 📊 Dataset

- **Source:** [SMS Spam Collection Dataset – Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 SMS messages
- **Classes:** Ham (4,825) and Spam (747)
- **Imbalance ratio:** ~87% Ham / ~13% Spam

---

## 🏗️ Project Structure

```
spam_classifier/
│
├── spam_classifier.py       # Main ML pipeline
├── spam.tsv                 # Dataset
├── README.md                # This file
└── plots/
    ├── eda.png              # Exploratory data analysis
    ├── evaluation.png       # Model comparison + confusion matrix
    ├── roc_curve.png        # ROC AUC curve
    └── top_words.png        # Most predictive words
```

---

## ⚙️ Pipeline Steps

### 1. Data Loading & EDA
- Load the tab-separated dataset
- Analyse class distribution, message lengths, and word counts

### 2. Text Preprocessing
- Lowercase conversion
- Remove punctuation and special characters (regex)
- Remove English stopwords (NLTK)
- Strip short/irrelevant tokens

### 3. Feature Extraction
- **TF-IDF Vectorizer** with bigrams (`ngram_range=(1,2)`, `max_features=8000`)
- Captures both individual words and two-word phrases

### 4. Model Training (4 Algorithms)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 96.41% | 100% | 73.15% | 84.50% |
| Logistic Regression | 96.95% | 100% | 77.18% | 87.12% |
| **Linear SVM** ⭐ | **98.39%** | **97.81%** | **89.93%** | **93.71%** |
| Random Forest | 97.31% | 100% | 79.87% | 88.81% |

### 5. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curve
- Cross-validation

---

## 🏆 Best Model — Linear SVM

```
              precision    recall  f1-score
Ham              0.98      1.00      0.99
Spam             0.98      0.90      0.94
Accuracy:                            0.98
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/spam-classifier.git
cd spam-classifier

# 2. Install dependencies
pip install pandas scikit-learn matplotlib seaborn nltk

# 3. Run the classifier
python spam_classifier.py
```

---

## 🔍 Sample Predictions

| Message | Prediction |
|---|---|
| "FREE prize! Call now to claim your £1000 reward." | 🚨 SPAM (83.8%) |
| "Hey, are you coming to the party tonight?" | ✅ HAM (97.8%) |
| "Congratulations! You've won a free iPhone." | 🚨 SPAM (68.9%) |
| "Can you pick up milk on your way home?" | ✅ HAM (97.5%) |

---

## 📦 Dependencies

```
pandas
scikit-learn
matplotlib
seaborn
nltk
numpy
```

---

## 👤 Author

[Your Name] | Machine Learning Internship Assessment
