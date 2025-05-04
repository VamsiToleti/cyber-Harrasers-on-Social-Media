import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import string

# Load the model
model = joblib.load('model.pkl')

# Load the vectorizer used for text transformation
vectorizer = joblib.load('vectorizer.pkl')

# Define the feature extraction function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Map predictions to classes
class_mapping = {
    0: "Hate Speech",
    1: "Offensive",
    2: "Neither"
}

def predict_class(tweet, hate_speech, offensive_language, neither):
    # Preprocess the text
    tweet = preprocess_text(tweet)
    
    # Vectorize the text
    tweet_vectorized = vectorizer.transform([tweet])
    
    # Combine text features and numerical features
    numeric_features = [hate_speech, offensive_language, neither]
    X_combined = hstack([tweet_vectorized, pd.DataFrame([numeric_features])])
    
    # Make prediction
    prediction = model.predict(X_combined)
    return prediction[0]

# Streamlit UI
st.title("Tweet Classification App")

tweet = st.text_area("Enter tweet:")
hate_speech = 0
offensive_language = 0
neither = 0

if st.button("Predict"):
    result = predict_class(tweet, hate_speech, offensive_language, neither)
    st.write(f"Predicted Class: {class_mapping.get(result, 'Unknown')}")
