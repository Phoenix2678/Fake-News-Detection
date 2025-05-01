# predict.py

import pickle

# Load vectorizer and model
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("models/logistic_model.pkl", "rb"))  # or "rf_model.pkl"

def predict_news(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return "REAL" if prediction[0] == 1 else "FAKE"

# Example
if __name__ == "__main__":
    sample = input("Enter a news article text:\n")
    result = predict_news(sample)
    print(f"\n🧠 Prediction: {result}")
