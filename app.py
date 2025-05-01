from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer once when the app starts
model = pickle.load(open("models/logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    error_message = ""
    if request.method == "POST":
        news = request.form["news"]
        if news.strip() != "":
            try:
                # Vectorize the news article and predict
                vector = vectorizer.transform([news])
                result = model.predict(vector)
                prediction = "REAL" if result[0] == 1 else "FAKE"
            except Exception as e:
                error_message = f"Error occurred while processing: {str(e)}"
        else:
            error_message = "Please enter a valid news article."

    return render_template("index.html", prediction=prediction, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)
