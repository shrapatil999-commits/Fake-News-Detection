
import streamlit as st
import pickle
import re

model = pickle.load(open("fake_news_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

st.title("📰 Fake News Detection System")
st.write("Enter a news article to check whether it is real or fake.")

news = st.text_area("Enter News Article")

if st.button("Predict"):
    if news.strip():
        cleaned = clean_text(news)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        st.subheader("Prediction Result")

        # Update this mapping according to your dataset labels
        if prediction == 1:
            st.success("REAL NEWS")
        else:
            st.error("FAKE NEWS")
    else:
        st.warning("Please enter a news article.")
