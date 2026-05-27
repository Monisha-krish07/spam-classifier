import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

print("=" * 60)
print("  SMS SPAM CLASSIFIER")
print("=" * 60)

# Load dataset
df = pd.read_csv('spam.tsv', sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
df = df[df['label'].isin(['ham', 'spam'])].reset_index(drop=True)
print(f"Dataset loaded: {len(df)} messages")
print(df['label'].value_counts().to_string())

df['label_num']  = df['label'].map({'ham': 0, 'spam': 1})
df['msg_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))

os.makedirs('plots', exist_ok=True)

# EDA plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('SMS Spam Classifier - Exploratory Data Analysis', fontsize=14, fontweight='bold')
colors = ['#2ecc71', '#e74c3c']
counts = df['label'].value_counts()
axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Class Distribution', fontweight='bold')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
for label, color in zip(['ham', 'spam'], colors):
    axes[1].hist(df[df['label'] == label]['msg_length'], bins=40,
                 alpha=0.7, color=color, label=label, edgecolor='white')
axes[1].set_title('Message Length Distribution', fontweight='bold')
axes[1].set_xlabel('Character Count')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].set_xlim(0, 500)
spam_words = df[df['label'] == 'spam']['word_count']
ham_words  = df[df['label'] == 'ham']['word_count']
bp = axes[2].boxplot([ham_words, spam_words], tick_labels=['Ham', 'Spam'], patch_artist=True, notch=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[2].set_title('Word Count by Class', fontweight='bold')
axes[2].set_ylabel('Word Count')
axes[2].set_ylim(0, 80)
plt.tight_layout()
plt.savefig('plots/eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA plot saved")

# Preprocessing
STOP_WORDS = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)

df['clean_message'] = df['message'].apply(preprocess)
print("Text preprocessing complete")
print("Sample:", df['clean_message'].iloc[0][:80])

# Split
X = df['clean_message']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Split -> Train: {len(X_train)}  |  Test: {len(X_test)}")

# TF-IDF + Models
vectorizer    = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

models = {
    'Naive Bayes':         MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM':          LinearSVC(random_state=42, max_iter=2000),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
print("\n" + "=" * 60)
print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 60)
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, output_dict=True)
    acc  = accuracy_score(y_test, y_pred)
    prec = report['1']['precision']
    rec  = report['1']['recall']
    f1   = report['1']['f1-score']
    results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec,
                     'f1': f1, 'model': model, 'y_pred': y_pred}
    print(f"{name:<22} {acc:>8.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")
print("=" * 60)

best_name = max(results, key=lambda k: results[k]['f1'])
best = results[best_name]
print(f"\nBest Model: {best_name}  (F1 = {best['f1']:.4f})")
print(classification_report(y_test, best['y_pred'], target_names=['Ham', 'Spam']))

# Confusion matrix + comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
axes[0].set_title(f'Confusion Matrix ({best_name})', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')
metric_names = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(metric_names))
width = 0.18
palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metric_names]
    axes[1].bar(x + i * width, vals, width, label=name, color=palette[i], alpha=0.85)
axes[1].set_title('Model Comparison', fontweight='bold')
axes[1].set_xticks(x + width * 1.5)
axes[1].set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
axes[1].set_ylim(0.85, 1.01)
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig('plots/evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Evaluation plot saved")

# ROC curve
lr_model = results['Logistic Regression']['model']
y_prob = lr_model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('plots/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("ROC curve saved")

# Top words
feature_names = vectorizer.get_feature_names_out()
lr_coef = lr_model.coef_[0]
top_spam = pd.Series(lr_coef, index=feature_names).nlargest(15)
top_ham  = pd.Series(lr_coef, index=feature_names).nsmallest(15)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top_spam.sort_values().plot(kind='barh', ax=axes[0], color='#e74c3c', alpha=0.8)
axes[0].set_title('Top 15 Spam Words', fontweight='bold')
top_ham.sort_values(ascending=False).plot(kind='barh', ax=axes[1], color='#2ecc71', alpha=0.8)
axes[1].set_title('Top 15 Ham Words', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/top_words.png', dpi=150, bbox_inches='tight')
plt.close()
print("Top words plot saved")

# Demo predictions
def predict_message(text):
    clean = preprocess(text)
    tfidf = vectorizer.transform([clean])
    pred  = lr_model.predict(tfidf)[0]
    prob  = lr_model.predict_proba(tfidf)[0]
    return "SPAM" if pred == 1 else "HAM", prob[0], prob[1]

print("\n" + "=" * 60)
print("  DEMO PREDICTIONS")
print("=" * 60)
for msg in [
    "FREE prize! Call now to claim your reward. Limited time!",
    "Hey, are you coming to the party tonight?",
    "Congratulations! You have won a free iPhone. Click here.",
    "Can you pick up milk on your way home?",
]:
    label, p_ham, p_spam = predict_message(msg)
    print(f"MSG : {msg}")
    print(f"PRED: {label}  (Ham: {p_ham:.2%} | Spam: {p_spam:.2%})\n")

print("All done!")