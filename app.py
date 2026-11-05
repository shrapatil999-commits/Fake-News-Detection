
import streamlit as st
import pickle
import re
import pandas as pd
from datetime import datetime

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)


# ---------------- SESSION HISTORY ----------------

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------- TEXT CLEANING ----------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------- LABEL MAPPING ----------------

def get_label_name(label):
    if str(label).lower() in ["1", "real", "true"]:
        return "REAL NEWS"
    return "FAKE NEWS"


# ---------------- CLEAR INPUT ----------------

def clear_news():
    st.session_state.news_input = ""


# ---------------- MAIN WEBSITE ----------------

st.title("📰 Fake News Detection System")

st.write(
    "An AI-powered platform for analyzing "
    "news articles using Machine Learning and NLP."
)

st.divider()

news = st.text_area(
    "Enter News Article",
    height=200,
    key="news_input",
    placeholder="Paste your news article here..."
)

col1, col2 = st.columns(2)

with col1:
    predict = st.button(
        "🔍 Analyze News",
        use_container_width=True
    )

with col2:
    st.button(
        "🗑️ Clear",
        on_click=clear_news,
        use_container_width=True
    )


# ---------------- PREDICTION ----------------

if predict:

    if news.strip():

        cleaned = clean_text(news)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        predicted_name = get_label_name(prediction)
        confidence = max(probabilities) * 100

        st.session_state.last_analysis = {
            "news": news,
            "prediction": predicted_name,
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
            "classes": model.classes_.tolist(),
            "time": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            )
        }

        # Save history
        st.session_state.history.insert(0, {
            "Date": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),
            "News": news[:100],
            "Prediction": predicted_name,
            "Confidence": round(confidence, 2)
        })

    else:
        st.warning("Please enter a news article.")


# ---------------- DISPLAY ANALYSIS ----------------

if "last_analysis" in st.session_state:

    result = st.session_state.last_analysis

    st.divider()
    st.header("📊 Prediction Result")

    if result["prediction"] == "REAL NEWS":
        st.success("✅ REAL NEWS")
    else:
        st.error("❌ FAKE NEWS")

    st.metric(
        "Prediction Confidence",
        f"{result['confidence']:.2f}%"
    )

    # -------- FEATURE 1: PROBABILITY CHART --------

    st.subheader("📈 REAL vs FAKE Probability")

    probability_data = {}

    for label, probability in zip(
        result["classes"],
        result["probabilities"]
    ):
        probability_data[get_label_name(label)] = (
            probability * 100
        )

    chart_df = pd.DataFrame(
        {"Probability (%)": probability_data}
    )

    st.bar_chart(chart_df)

    st.caption(
        "These probabilities represent the model's "
        "estimated class probabilities, not verified truth."
    )

    # -------- FEATURE 2: IMPORTANT WORD ANALYSIS --------

    st.subheader("🔎 Important Word Analysis")

    if len(model.classes_) == 2 and hasattr(model, "coef_"):

        feature_names = tfidf.get_feature_names_out()
        text_values = vector.toarray()[0]

        coefficients = model.coef_[0]

        contributions = text_values * coefficients

        positive_indices = contributions.argsort()[::-1]
        negative_indices = contributions.argsort()

        positive_words = []
        negative_words = []

        for index in positive_indices:
            if contributions[index] > 0:
                positive_words.append({
                    "Word": feature_names[index],
                    "Influence": round(
                        float(contributions[index]), 4
                    )
                })

        for index in negative_indices:
            if contributions[index] < 0:
                negative_words.append({
                    "Word": feature_names[index],
                    "Influence": round(
                        float(abs(contributions[index])), 4
                    )
                })

        positive_class = model.classes_[1]

        if get_label_name(positive_class) == "REAL NEWS":
            positive_title = "Words supporting REAL prediction"
            negative_title = "Words supporting FAKE prediction"
        else:
            positive_title = "Words supporting FAKE prediction"
            negative_title = "Words supporting REAL prediction"

        with st.expander(positive_title):
            if positive_words:
                st.dataframe(
                    pd.DataFrame(positive_words[:10]),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.write("No strong positive word contributions.")

        with st.expander(negative_title):
            if negative_words:
                st.dataframe(
                    pd.DataFrame(negative_words[:10]),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.write("No strong negative word contributions.")

    else:
        st.info(
            "Word contribution analysis currently supports "
            "binary Logistic Regression models."
        )


# ---------------- FEATURE 3: PREDICTION HISTORY ----------------

st.divider()
st.header("📚 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("Clear Prediction History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("Your analyzed news articles will appear here.")


# ---------------- ABOUT PROJECT ----------------

with st.expander("📌 About Project"):

    st.write("""
    Fake News Detection is a Machine Learning project
    that classifies news articles as REAL or FAKE.

    Technologies:
    - Python
    - Natural Language Processing
    - TF-IDF Vectorization
    - Logistic Regression
    - Streamlit
    """)


# ---------------- HOW IT WORKS ----------------

with st.expander("⚙️ How It Works"):

    st.write("""
    1. Enter a news article.
    2. Clean and preprocess the text.
    3. Convert text into numerical features using TF-IDF.
    4. Classify the article using Logistic Regression.
    5. Display prediction and probability analysis.
    6. Store the result in prediction history.
    """)

st.caption(
    "Developed using Python, NLP and Machine Learning."
)


