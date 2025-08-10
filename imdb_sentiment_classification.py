# Sentiment Classification of Movie Reviews with NLP Feature Engineering
# This script shows my progress in NLP + ML since my last project. 
# Compared to my previous fake-news classifier, here I add sentiment scores, topic modeling, and gradient boosting.

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from textblob import TextBlob

# Download necessary NLTK packages
nltk.download('punkt')

# -----------------------------
# 1. Load IMDB dataset (binary sentiment: positive / negative)
# -----------------------------
df = pd.read_csv("编程作品/IMDB Dataset.csv")  # Should have 'review' and 'sentiment' columns
print(df.head())

# Encode target: positive -> 1, negative -> 0
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# -----------------------------
# 2. Basic text cleaning
# -----------------------------
def clean_text(text):
    # Simple lowercase + tokenization
    tokens = nltk.word_tokenize(str(text).lower())
    # Keep only words (remove punctuation, numbers)
    words = [word for word in tokens if word.isalpha()]
    return " ".join(words)

df['clean_review'] = df['review'].apply(clean_text)

# -----------------------------
# 3. Sentiment score feature (using TextBlob)
# -----------------------------
df['polarity'] = df['clean_review'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['clean_review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# -----------------------------
# 4. TF-IDF vectorization
# -----------------------------
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['clean_review']).toarray()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])

# -----------------------------
# 5. Topic modeling features (LDA on TF-IDF)
# -----------------------------
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_topics = lda.fit_transform(tfidf_matrix)
lda_df = pd.DataFrame(lda_topics, columns=[f"topic_{i}" for i in range(lda_topics.shape[1])])

# -----------------------------
# 6. Combine all features
# -----------------------------
X = pd.concat([tfidf_df, lda_df, df[['polarity', 'subjectivity']]], axis=1)
y = df['label']

# -----------------------------
# 7. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 8. Train models
# -----------------------------
# Logistic Regression (baseline)
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
print("Logistic Regression:")
print(classification_report(y_test, lr.predict(X_test)))

# Gradient Boosting (new to me compared to last project)
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
print("Gradient Boosting:")
print(classification_report(y_test, gb.predict(X_test)))

# -----------------------------
# 9. Confusion matrix for Gradient Boosting
# -----------------------------
cm = confusion_matrix(y_test, gb.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Gradient Boosting - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 10. Feature importance from Gradient Boosting
# -----------------------------
importances = gb.feature_importances_
# Grab top 10 important features
indices = np.argsort(importances)[-10:]
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title("Top 10 Important Features (GB)")
plt.show()