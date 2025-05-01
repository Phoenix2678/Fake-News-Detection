# train_models.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Load cleaned data
df = pd.read_csv("data/cleaned_news.csv")

# Map label text to numeric
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# Split data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf_model.fit(X_train_vec, y_train)

# Evaluate models on test data
lr_pred = lr_model.predict(X_test_vec)
rf_pred = rf_model.predict(X_test_vec)

# Print Classification Reports
print("\n📊 Logistic Regression:")
print(classification_report(y_test, lr_pred))

print("\n🌲 Random Forest:")
print(classification_report(y_test, rf_pred))

# Print accuracy scores
lr_accuracy = accuracy_score(y_test, lr_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"\n🔍 Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"🔍 Random Forest Accuracy: {rf_accuracy:.4f}")

# Perform Cross-validation for both models (for more robust evaluation)
lr_cv_score = cross_val_score(lr_model, X_train_vec, y_train, cv=5)
rf_cv_score = cross_val_score(rf_model, X_train_vec, y_train, cv=5)

print(f"\n🔄 Logistic Regression Cross-Validation Score (5-fold): {lr_cv_score.mean():.4f}")
print(f"🔄 Random Forest Cross-Validation Score (5-fold): {rf_cv_score.mean():.4f}")

# Save models
os.makedirs("models", exist_ok=True)
pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))
pickle.dump(lr_model, open("models/logistic_model.pkl", "wb"))
pickle.dump(rf_model, open("models/rf_model.pkl", "wb"))

print("\n✅ Models and vectorizer saved in 'models/' folder.")
