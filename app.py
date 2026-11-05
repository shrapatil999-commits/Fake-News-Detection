
import streamlit as st
import pickle
import re

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Load trained model
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Clear input function
def clear_news():
    st.session_state.news_input = ""


# Main heading
st.title("📰 Fake News Detection System")

st.write("Machine Learning Based Fake News Classification")

st.info("Enter a news article to check its authenticity.")

# News input
news = st.text_area(
    "Enter News Article",
    height=200,
    key="news_input"
)

# Buttons
col1, col2 = st.columns(2)

with col1:
    predict = st.button("🔍 Detect News", use_container_width=True)

with col2:
    st.button(
        "🗑️ Clear",
        on_click=clear_news,
        use_container_width=True
    )

# Prediction
if predict:
    if news.strip():

        cleaned = clean_text(news)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)[0]
        confidence = max(probabilities) * 100

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    else:
        st.warning("Please enter a news article.")

# About Project
with st.expander("📌 About Project"):
    st.write(
        "This project uses Machine Learning and "
        "Natural Language Processing to classify "
        "news articles as REAL or FAKE."
    )

# How it works
with st.expander("⚙️ How It Works"):
    st.write("""
    1. User enters a news article.
    2. Text preprocessing cleans the article.
    3. TF-IDF converts text into numerical features.
    4. Logistic Regression predicts the category.
    5. The application displays the prediction
       and confidence score.
    """)

st.caption("Developed using Python, NLP and Machine Learning.")


