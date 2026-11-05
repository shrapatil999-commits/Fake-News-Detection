import streamlit as st
import pickle
import re

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

model = pickle.load(open("fake_news_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

st.title("📰 Fake News Detection System")
st.write("Machine Learning based Fake News Classification")

st.info("Enter a news article below to predict its authenticity.")

news = st.text_area("Enter News Article", height=200)

if st.button("🔍 Detect News"):
    if news.strip():
        cleaned = clean_text(news)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

probabilities = model.predict_proba(vector)[0]
confidence = max(probabilities) * 100

st.subheader("Prediction Result")

if prediction == 1:
    st.success("REAL NEWS")
else:
    st.error("FAKE NEWS")

st.metric("Prediction Confidence", f"{confidence:.2f}%")
