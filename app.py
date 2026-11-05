
import streamlit as st
import pickle
import re
import pandas as pd
from datetime import datetime

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_files():
    with open("fake_news_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("tfidf_vectorizer.pkl", "rb") as file:
        tfidf = pickle.load(file)

    return model, tfidf


model, tfidf = load_files()

# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "news_input" not in st.session_state:
    st.session_state.news_input = ""


# ---------------- TEXT PREPROCESSING ----------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------- LABEL CONVERSION ----------------

def get_label_name(label):
    label_text = str(label).strip().lower()

    if label_text in ["1", "real", "true"]:
        return "REAL"

    if label_text in ["0", "fake", "false"]:
        return "FAKE"

    return label_text.upper()


# ---------------- CLEAR INPUT ----------------

def clear_input():
    st.session_state.news_input = ""
    st.session_state.last_result = None


# ---------------- APP TITLE ----------------

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article or headline below to check "
    "whether it is likely to be REAL or FAKE."
)

st.divider()

# ---------------- NEWS INPUT ----------------

news = st.text_area(
    "Enter News Text:",
    height=180,
    placeholder="Paste your news article here...",
    key="news_input"
)

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button(
        "🔍 Predict News",
        use_container_width=True
    )

with col2:
    st.button(
        "🧹 Clear Input",
        use_container_width=True,
        on_click=clear_input
    )


# ---------------- PREDICTION ----------------

if predict_button:

    if not news.strip():
        st.warning("Please enter some news text first.")

    else:
        cleaned_news = clean_text(news)

        if not cleaned_news:
            st.warning("Please enter valid news text.")

        else:
            news_vector = tfidf.transform([cleaned_news])

            prediction = model.predict(news_vector)[0]
            probabilities = model.predict_proba(news_vector)[0]

            prediction_name = get_label_name(prediction)

            confidence = max(probabilities) * 100

            probability_data = {}

            for class_label, probability in zip(
                model.classes_, probabilities
            ):
                display_label = get_label_name(class_label)
                probability_data[display_label] = round(
                    probability * 100, 2
                )

            # Save the latest result
            st.session_state.last_result = {
                "news": news,
                "prediction": prediction_name,
                "confidence": round(confidence, 2),
                "probabilities": probability_data
            }

            # Add prediction to history
            history_item = {
                "Date & Time": datetime.now().strftime(
                    "%d-%m-%Y %I:%M:%S %p"
                ),
                "News": news[:100] + ("..." if len(news) > 100 else ""),
                "Prediction": prediction_name,
                "Confidence": f"{confidence:.2f}%"
            }

            st.session_state.history.insert(0, history_item)


# ---------------- DISPLAY RESULT ----------------

result = st.session_state.last_result

if result is not None:

    st.divider()
    st.subheader("📊 Prediction Result")

    if result["prediction"] == "REAL":
        st.success("This news is predicted to be REAL.")

    elif result["prediction"] == "FAKE":
        st.error("This news is predicted to be FAKE.")

    else:
        st.info(f"Prediction: {result['prediction']}")

    st.metric(
        "Model Confidence",
        f"{result['confidence']:.2f}%"
    )

    st.caption(
        "Confidence is the model's estimated probability, "
        "not proof that the news is true or false."
    )

    # ---------------- PROBABILITY CHART ----------------

    st.subheader("📈 REAL vs FAKE Probability")

    chart_data = pd.DataFrame(
        {"Probability (%)": result["probabilities"]}
    )

    st.bar_chart(chart_data)

    st.write("**Probability Details:**")

    for label, probability in result["probabilities"].items():
        st.write(f"{label}: {probability:.2f}%")


# ---------------- PREDICTION HISTORY ----------------

st.divider()
st.subheader("🕘 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("Your prediction history will appear here.")


# ---------------- ABOUT PROJECT ----------------

st.divider()

with st.expander("ℹ️ About Project"):
    st.write(
        """
        This Fake News Detection System uses Machine Learning
        and Natural Language Processing (NLP).

        The model uses TF-IDF for text feature extraction
        and Logistic Regression for classification.
        """
    )

with st.expander("⚙️ How It Works"):
    st.write(
        """
        1. Enter a news article.
        2. The text is cleaned and preprocessed.
        3. TF-IDF converts the text into numerical features.
        4. Logistic Regression predicts the news category.
        5. The result and probability chart are displayed.
        """
    )



