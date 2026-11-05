
import streamlit as st
import pickle
import re
import pandas as pd
import html
import io

from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# ==========================================
# 2. THEME SETTINGS
# ==========================================

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

st.sidebar.title("⚙️ Settings")

theme = st.sidebar.radio(
    "Choose Appearance",
    ["Light", "Dark"],
    index=["Light", "Dark"].index(
        st.session_state.theme
    )
)

st.session_state.theme = theme

if theme == "Dark":

    st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: white;
        }

        [data-testid="stSidebar"] {
            background-color: #1e1e1e;
        }

        h1, h2, h3, p, label {
            color: white !important;
        }

        .stTextArea textarea {
            background-color: #292929;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            color: black;
        }

        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        </style>
    """, unsafe_allow_html=True)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_files():

    with open("fake_news_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("tfidf_vectorizer.pkl", "rb") as file:
        tfidf = pickle.load(file)

    return model, tfidf


model, tfidf = load_files()


# ==========================================
# 4. SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "news_input" not in st.session_state:
    st.session_state.news_input = ""


# ==========================================
# 5. TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================
# 6. LABEL CONVERSION
# ==========================================

def get_label_name(label):

    label_text = str(label).strip().lower()

    if label_text in ["1", "real", "true"]:
        return "REAL"

    if label_text in ["0", "fake", "false"]:
        return "FAKE"

    return label_text.upper()


# ==========================================
# 7. PDF REPORT GENERATION
# ==========================================

def create_pdf_report(result):

    buffer = io.BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    story = []

    # Report Title

    story.append(
        Paragraph(
            "<b>Fake News Detection Report</b>",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 20))

    # Date and Time

    report_time = result.get(
        "datetime",
        "Not available"
    )

    story.append(
        Paragraph(
            "<b>Date &amp; Time:</b> "
            + html.escape(str(report_time)),
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 20))

    # Prediction Result

    story.append(
        Paragraph(
            "<b>Prediction Result</b>",
            styles["Heading2"]
        )
    )

    prediction = html.escape(
        str(result["prediction"])
    )

    story.append(
        Paragraph(
            f"<b>News Classification:</b> {prediction}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 10))

    # Confidence Score

    story.append(
        Paragraph(
            f"<b>Model Confidence:</b> "
            f"{result['confidence']:.2f}%",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 20))

    # Probability Table

    story.append(
        Paragraph(
            "<b>REAL vs FAKE Probability</b>",
            styles["Heading2"]
        )
    )

    table_data = [
        ["Category", "Probability"]
    ]

    for label, probability in result["probabilities"].items():

        table_data.append([
            html.escape(str(label)),
            f"{probability:.2f}%"
        ])

    probability_table = Table(
        table_data,
        colWidths=[220, 180]
    )

    probability_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.darkblue
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            )
        ])
    )

    story.append(probability_table)

    story.append(Spacer(1, 20))

    # Original News Article

    story.append(
        Paragraph(
            "<b>Original News Article</b>",
            styles["Heading2"]
        )
    )

    safe_news = html.escape(
        str(result["news"])
    )

    safe_news = safe_news.replace(
        "\n",
        "<br/>"
    )

    # Handle characters unsupported by standard PDF fonts
    safe_news = (
        safe_news.encode("latin-1", "replace")
        .decode("latin-1")
    )

    story.append(
        Paragraph(
            safe_news,
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 25))

    # Disclaimer

    story.append(
        Paragraph(
            "<i>Disclaimer: This report contains predictions "
            "generated by a machine learning model. "
            "Model confidence is not proof of factual accuracy.</i>",
            styles["Italic"]
        )
    )

    # Generate PDF

    pdf.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ==========================================
# 8. CLEAR INPUT
# ==========================================

def clear_input():

    st.session_state.news_input = ""

    st.session_state.last_result = None


# ==========================================
# 9. APP TITLE
# ==========================================

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article or headline below to check "
    "whether it is likely to be REAL or FAKE."
)

st.divider()


# ==========================================
# 10. NEWS INPUT
# ==========================================

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


# ==========================================
# 11. PREDICTION
# ==========================================

if predict_button:

    if not news.strip():

        st.warning(
            "Please enter some news text first."
        )

    else:

        cleaned_news = clean_text(news)

        if not cleaned_news:

            st.warning(
                "Please enter valid news text."
            )

        else:

            news_vector = tfidf.transform(
                [cleaned_news]
            )

            prediction = model.predict(
                news_vector
            )[0]

            probabilities = model.predict_proba(
                news_vector
            )[0]

            prediction_name = get_label_name(
                prediction
            )

            confidence = max(probabilities) * 100

            probability_data = {}

            for class_label, probability in zip(
                model.classes_,
                probabilities
            ):

                display_label = get_label_name(
                    class_label
                )

                probability_data[display_label] = round(
                    probability * 100,
                    2
                )

            # Save latest prediction

            prediction_time = datetime.now().strftime(
                "%d-%m-%Y %I:%M:%S %p"
            )

            st.session_state.last_result = {

                "news": news,

                "prediction": prediction_name,

                "confidence": round(
                    confidence,
                    2
                ),

                "probabilities": probability_data,

                "datetime": prediction_time
            }

            # Save prediction history

            history_item = {

                "Date & Time": prediction_time,

                "News": news[:100] + (
                    "..." if len(news) > 100 else ""
                ),

                "Prediction": prediction_name,

                "Confidence": f"{confidence:.2f}%"
            }

            st.session_state.history.insert(
                0,
                history_item
            )


# ==========================================
# 12. DISPLAY PREDICTION RESULT
# ==========================================

result = st.session_state.last_result

if result is not None:

    st.divider()

    st.subheader("📊 Prediction Result")

    if result["prediction"] == "REAL":

        st.success(
            "This news is predicted to be REAL."
        )

    elif result["prediction"] == "FAKE":

        st.error(
            "This news is predicted to be FAKE."
        )

    else:

        st.info(
            f"Prediction: {result['prediction']}"
        )

    # Confidence Score

    st.metric(
        "Model Confidence",
        f"{result['confidence']:.2f}%"
    )

    st.caption(
        "Confidence is the model's estimated probability, "
        "not proof that the news is true or false."
    )

    # Probability Chart

    st.subheader(
        "📈 REAL vs FAKE Probability"
    )

    chart_data = pd.DataFrame(
        {"Probability (%)": result["probabilities"]}
    )

    st.bar_chart(chart_data)

    st.write("**Probability Details:**")

    for label, probability in result["probabilities"].items():

        st.write(
            f"{label}: {probability:.2f}%"
        )


# ==========================================
# 13. DOWNLOAD PDF REPORT
# ==========================================

if result is not None:

    st.divider()

    st.subheader(
        "📄 Download Prediction Report"
    )

    pdf_file = create_pdf_report(result)

    st.download_button(
        label="📥 Download Report as PDF",
        data=pdf_file,
        file_name="Fake_News_Prediction_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )


# ==========================================
# 14. PREDICTION HISTORY
# ==========================================

st.divider()

st.subheader("🕘 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.history = []

        st.rerun()

else:

    st.info(
        "Your prediction history will appear here."
    )


# ==========================================
# 15. ABOUT PROJECT
# ==========================================

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


# ==========================================
# 16. HOW IT WORKS
# ==========================================

with st.expander("⚙️ How It Works"):

    st.write(
        """
        1. Enter a news article.
        2. The text is cleaned and preprocessed.
        3. TF-IDF converts the text into numerical features.
        4. Logistic Regression predicts the news category.
        5. The result and probability chart are displayed.
        6. Download the prediction report as a PDF.
        """
    )
