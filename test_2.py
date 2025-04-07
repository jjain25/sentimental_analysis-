import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import docx
import PyPDF2
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Streamlit Setup ---
st.set_page_config(layout="wide", page_title="📊 Unified Financial Sentiment Dashboard")
st.title("📈 Unified Financial Sentiment Analysis Dashboard")

# --- NLTK and VADER Setup ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# --- Load FinBERT ---
# Try this once with internet ON
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", cache_dir="./model", local_files_only=False)
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", cache_dir="./model", local_files_only=False)
tokenizer.save_pretrained("./models/finbert")
model.save_pretrained("./models/finbert")


# --- Load Loughran-McDonald Dictionary ---
lmd_df = pd.read_csv("D:/jinay/Loughran-McDonald_MasterDictionary_1993-2024.csv")
positive_words = set(lmd_df[lmd_df['Positive'] > 0]['Word'].str.lower())
negative_words = set(lmd_df[lmd_df['Negative'] > 0]['Word'].str.lower())


# --- Load FinBERT ---
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", cache_dir="./model", local_files_only=False)
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", cache_dir="./model", local_files_only=False)

    return tokenizer, model

finbert_tokenizer, finbert_model = load_finbert()

# --- Load Loughran-McDonald Dictionary ---
@st.cache_data
def load_lm_dict():
    lm_df = pd.read_csv("D:/jinay/Loughran-McDonald_MasterDictionary_1993-2024.csv")
    positive = set(lm_df[lm_df["Positive"] > 0]["Word"].str.lower())
    negative = set(lm_df[lm_df["Negative"] > 0]["Word"].str.lower())
    return positive, negative

lm_positive, lm_negative = load_lm_dict()

# --- File/Text Handling ---
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    else:
        return file.getvalue().decode("utf-8")

# --- Analysis Methods ---
def analyze_vader(text):
    return sia.polarity_scores(text)

def analyze_finbert(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = finbert_model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
    return {"Positive": scores[0], "Neutral": scores[1], "Negative": scores[2]}

def analyze_textblob(text):
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}

def lmd_sentiment_score(text):
    words = text.lower().split()
    pos = sum(word in positive_words for word in words)
    neg = sum(word in negative_words for word in words)
    total = pos + neg
    if total == 0:
        return {"positive_ratio": 0, "negative_ratio": 0, "sentiment": "Neutral"}
    sentiment = "Positive" if pos > neg else "Negative" if neg > pos else "Neutral"
    return {"positive_ratio": pos / total, "negative_ratio": neg / total, "sentiment": sentiment}

def analyze_lm_dict(text):
    words = [w.lower() for w in text.split()]
    pos_count = sum(1 for word in words if word in lm_positive)
    neg_count = sum(1 for word in words if word in lm_negative)
    total = pos_count + neg_count
    sentiment = "Neutral"
    if pos_count > neg_count:
        sentiment = "Positive"
    elif neg_count > pos_count:
        sentiment = "Negative"
    return {
        "Positive Words": pos_count,
        "Negative Words": neg_count,
        "Total Matched": total,
        "Sentiment": sentiment
    }

def analyze_subjectivity(text):
    blob = TextBlob(text)
    return {"Polarity": blob.sentiment.polarity, "Subjectivity": blob.sentiment.subjectivity}

def classify_risk(val):
    if val < -0.5:
        return "⚠️ High Risk"
    elif -0.5 <= val <= -0.2:
        return "⚠️ Moderate Risk"
    elif -0.2 <= val <= 0.2:
        return "🟡 Neutral"
    else:
        return "✅ Stable"

def credit_rating(risk_level):
    return {
        "High Risk": "BB or lower",
        "Moderate Risk": "BBB",
        "Neutral": "A",
        "Stable": "AA or AAA"
    }.get(risk_level, "Not Rated")

def classify_sentence(sentence):
    vader_score = sia.polarity_scores(sentence)['compound']
    if vader_score >= 0.05:
        return "Positive"
    elif vader_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# --- Sidebar Input ---
uploaded = st.sidebar.file_uploader("📂 Upload Document", type=["pdf", "docx", "txt"])
text_input = st.sidebar.text_area("Or paste financial text here:", height=150)
keyword = st.sidebar.text_input("🔍 Optional: Filter sentences by keyword")

text = extract_text(uploaded) if uploaded else text_input
if keyword and text:
    text = ". ".join([s for s in text.split(". ") if keyword.lower() in s.lower()])

# --- Display ---
if text:
    st.subheader("📄 Extracted Text")
    st.text_area("Preview", text[:2000] + "..." if len(text) > 2000 else text, height=200)

    st.subheader("📈 Sentiment Analysis Results")
    vader_res = analyze_vader(text)
    finbert_res = analyze_finbert(text)
    lm_res = analyze_lm_dict(text)
    textblob_res = analyze_textblob(text)
    subj_res = analyze_subjectivity(text)
    
    risk_level = classify_risk(vader_res['compound'])
    rating = credit_rating(risk_level)

    result_df = pd.DataFrame([
        {
            "Model": "VADER",
            "Positive": vader_res["pos"],
            "Neutral": vader_res["neu"],
            "Negative": vader_res["neg"],
            "Score": vader_res["compound"],
            "Sentiment": "Positive" if vader_res["compound"] > 0.05 else "Negative" if vader_res["compound"] < -0.05 else "Neutral",
            "Risk": classify_risk(vader_res["compound"]),
            "Details": f"Pos: {vader_res['pos']}, Neu: {vader_res['neu']}, Neg: {vader_res['neg']}",
            "Interpretation": risk_level
        },
        {
            "Model": "FinBERT",
            "Positive": finbert_res["Positive"],
            "Neutral": finbert_res["Neutral"],
            "Negative": finbert_res["Negative"],
            "Score": finbert_res["Positive"] - finbert_res["Negative"],
            "Sentiment": max(finbert_res, key=finbert_res.get).capitalize(),
            "Risk": classify_risk(finbert_res["Positive"] - finbert_res["Negative"]),
            "Details": f"Pos: {finbert_res['Positive']:.2f}, Neu: {finbert_res['Neutral']:.2f}, Neg: {finbert_res['Negative']:.2f}",
            "Interpretation": "Probability-based sentiment"
        },
        {
            "Model": "Loughran-McDonald",
            "Positive": lm_res["Positive Words"],
            "Neutral": None,
            "Negative": lm_res["Negative Words"],
            "Score": lm_res["Positive Words"] - lm_res["Negative Words"],
            "Sentiment": lm_res["Sentiment"],
            "Risk": "⚠️ Risk" if lm_res["Negative Words"] > lm_res["Positive Words"] else "✅ Low Risk",
            "Details": f"{lm_res['Positive Words']} Positive, {lm_res['Negative Words']} Negative",
            "Interpretation": lm_res["Sentiment"]
        },
        {
            "Model": "TextBlob",
            "Positive": None,
            "Neutral": None,
            "Negative": None,
            "Score": subj_res["Polarity"],
            "Sentiment": "Positive" if subj_res["Polarity"] > 0 else "Negative" if subj_res["Polarity"] < 0 else "Neutral",
            "Risk": classify_risk(subj_res["Polarity"]),
            "Details": f"Polarity: {subj_res['Polarity']:.2f}, Subjectivity: {subj_res['Subjectivity']:.2f}",
            "Interpretation": f"Polarity={subj_res['Polarity']:.2f}, Subjectivity={subj_res['Subjectivity']:.2f}"
        }
    ])

    st.dataframe(result_df, use_container_width=True)

    
    st.markdown(f"### 🧾 Risk Classification: `{risk_level}`")
    st.markdown(f"### 🏦 Implied Credit Rating: `{rating}`")

    # --- Charts ---
    st.subheader("📊 Visualizations with Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.bar(["Positive", "Neutral", "Negative"],
               [finbert_res["Positive"], finbert_res["Neutral"], finbert_res["Negative"]],
               color=["green", "gray", "red"])
        ax.set_title("FinBERT Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.bar(["Compound Score"], [vader_res["compound"]], color='blue')
        ax.axhline(0.05, color='green', linestyle='--', label='Positive Threshold')
        ax.axhline(-0.05, color='red', linestyle='--', label='Negative Threshold')
        ax.set_ylim(-1, 1)
        ax.legend()
        ax.set_title("VADER Compound Score")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots()
        ax.bar(["Polarity", "Subjectivity"],
               [subj_res["Polarity"], subj_res["Subjectivity"]],
               color=["blue", "orange"])
        ax.set_title("TextBlob Analysis")
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots()
        ax.bar(["Positive Ratio", "Negative Ratio"],
               [lm_res["Positive Words"], lm_res["Negative Words"]],
               color=["green", "red"])
        ax.set_title("Loughran-McDonald Word Count")
        st.pyplot(fig)

    st.subheader("📘 Interpretation Guide")
    st.markdown("""
    - **VADER**: Compound score ranges from -1 to +1. 
        - Positive sentiment: > 0.05
        - Negative sentiment: < -0.05
        - Neutral sentiment: between -0.05 and 0.05
        - Implies a general sentiment tilt based on social media-like text features.

    - **FinBERT**: Specialized for financial domain.
        - Returns probabilities for: Positive, Neutral, Negative.
        - Highest probability determines overall sentiment.
        - Implies how market or financial language typically signals sentiment.

    - **Loughran-McDonald**:
        - Uses predefined financial-positive and financial-negative word lists.
        - Positive/Negative ratios give insight into tone of language in a financial context.
        - Useful for analyzing filings, statements, and disclosures.

    - **TextBlob**:
        - **Polarity**: -1 (most negative) to +1 (most positive)
        - **Subjectivity**: 0 (very objective) to 1 (very subjective)
        - High subjectivity implies opinionated text, while low suggests factual.
        - Helps understand tone and confidence level in narrative.
    """)

else:
    st.info("📌 Please upload a document or input text to analyze.")
