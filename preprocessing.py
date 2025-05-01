# preprocessing.py

import pandas as pd
import string
import os

# Load both fake and real datasets
fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")

# Add labels
fake_df["label"] = "FAKE"
real_df["label"] = "REAL"

# Combine
df = pd.concat([fake_df, real_df], axis=0).sample(frac=1, random_state=42)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

df['text'] = df['text'].apply(clean_text)

# Keep only necessary columns
df = df[['text', 'label']]

# Save cleaned data
os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned_news.csv", index=False)

print("✅ Combined and cleaned dataset saved as cleaned_news.csv")
