# Fake News Detection

## Overview
This project implements a machine learning system to detect fake news articles based on their content. Using natural language processing techniques and classification algorithms, the system analyzes text to determine the likelihood of an article being fake or real news.

## Features
- Text preprocessing and cleaning
- TF-IDF vectorization for feature extraction
- Multiple classification models:
  - Logistic Regression
  - Random Forest
- Web application for real-time fake news detection
- High accuracy in classifying news articles

## Dataset
The project uses a dataset from kaggle containing labeled news articles with 'REAL' and 'FAKE' classifications. The dataset includes:
- Title of the news article
- Text content
- Label indicating real or fake

## Model Performance
Our models achieved the following accuracy scores:
- Logistic Regression: 98.70%
- Random Forest: 98.50%

## Technical Implementation
- Python for data preprocessing and model training
- Scikit-learn for machine learning models
- Flask for web application development
- NLTK for natural language processing
- Deployed on Render for public access

## Deployment
The application is deployed and accessible at: [https://fake-news-detection-y6tw.onrender.com/](https://fake-news-detection-y6tw.onrender.com/)

## Usage
To use the deployed application:
1. Navigate to the URL
2. Input a news article text or paste the full content
3. Click "Predict" to get the classification result

## Future Improvements
- Implement additional machine learning models
- Enhance the UI/UX of the web application
- Add support for URL input to directly analyze online news articles

## License
This project is open source and available under the [MIT License](LICENSE).
