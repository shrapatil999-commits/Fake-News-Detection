# Detection of Misleading News Articles

A beginner-friendly AIML mini project using TF-IDF and Logistic Regression to classify article text as `FAKE` or `REAL`.

## 1. Project structure

```text
misleading_news_detection/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── (put Fake.csv and True.csv here)
└── models/
    └── (created after training)
```

## 2. Dataset

Download the **ISOT Fake News Dataset** from its original Kaggle listing:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Extract the archive and copy `Fake.csv` and `True.csv` into the `data/` folder. Check the dataset's license and terms before redistribution. The code expects a `text` column in each CSV.

## 3. Install and run

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train.py
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## 4. How it works

1. Load labeled real and fake article text from CSV files.
2. Clean text by lowercasing, removing URLs, punctuation and extra spaces.
3. Split the dataset into training (80%) and testing (20%) sets.
4. Convert text into numerical features using TF-IDF.
5. Train a Logistic Regression classifier.
6. Evaluate using accuracy, precision, recall, F1-score and a confusion matrix.
7. Save the trained pipeline and load it in the Streamlit app.

## 5. Important limitations

- The model learns patterns in its training dataset; it does not verify claims against trusted sources.
- Dataset bias, outdated examples, political or topic imbalance, and duplicate articles can affect results.
- A confidence score is not proof of truth.
- Do not use this prototype for high-stakes decisions or to label a person/source as deceptive.
- A stronger version should use source verification, claim extraction, evidence retrieval, human review and tests on recent unseen articles.

## 6. Suggested extensions

- Compare Multinomial Naive Bayes, Linear SVM and Logistic Regression.
- Add a confusion-matrix visualization.
- Add a URL/article-fetching module only where permitted by site terms.
- Add multilingual datasets and language-specific preprocessing.
- Add evidence links from reputable fact-checking organizations.
