import streamlit as st
from summarizer import generate_summary, textrank_summary, compare_summaries
from emotion_detector import detect_emotions
from utils import generate_wordcloud, plot_emotion_bar, text_stats, extract_keywords
import pandas as pd

st.set_page_config(page_title="Summify+ Pro", page_icon="🪶", layout="wide")
st.title("🪶 Summify+ Pro — Emotion-Aware Text Summarizer")
st.write("Summarize long text, compare models, extract keywords, and analyze emotions — with visuals.")

with st.sidebar:
    st.header("Options")
    model_opt = st.selectbox("Summarization model", ["BART (Abstractive)", "TextRank (Extractive)", "Compare Both"])
    show_wc = st.checkbox("Show Wordcloud for Summary", value=True)
    show_stats = st.checkbox("Show Text Stats", value=True)
    show_keywords = st.checkbox("Show Keywords (TF-IDF)", value=True)
    show_emotions = st.checkbox("Show Emotion Distribution", value=True)

uploaded_file = st.file_uploader("Upload a CSV with a 'text' column for batch processing (optional)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        st.info("Processing uploaded CSV...")
        df['Summary_BART'] = df['text'].apply(lambda t: generate_summary(t) if t and len(t.split())>30 else "Input too short for BART.")
        df['Summary_TextRank'] = df['text'].apply(lambda t: textrank_summary(t))
        emotions = df['text'].apply(lambda t: detect_emotions(t))
        df['Dominant_Emotion'] = emotions.apply(lambda x: x[0])
        df['Emotion_Confidence'] = emotions.apply(lambda x: x[1])
        st.write("Preview of processed results:")
        st.dataframe(df.head())
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download results CSV", csv_bytes, "summify_results.csv", "text/csv")

st.markdown("---")
text_input = st.text_area("Paste your text here (or use a sample from the dropdown):", height=300)
sample = st.selectbox("Or choose a sample text:", ["--none--", "Sample - News article", "Sample - Opinion piece"])
if sample != "--none--" and not text_input.strip():
    if sample == "Sample - News article":
        text_input = open("sample_texts.txt").read().split("===SAMPLE===")[1].strip()
    else:
        text_input = open("sample_texts.txt").read().split("===SAMPLE===")[2].strip()

if st.button("Run Analysis"):
    if not text_input.strip():
        st.warning("Please provide some text.")
    else:
        if show_stats:
            st.subheader("🔎 Text Statistics")
            text_stats(text_input)
        st.subheader("📝 Summary")
        if model_opt == "BART (Abstractive)":
            summary = generate_summary(text_input)
            st.write(summary)
        elif model_opt == "TextRank (Extractive)":
            summary = textrank_summary(text_input)
            st.write(summary)
        else:
            bart_sum = generate_summary(text_input)
            tr_sum = textrank_summary(text_input)
            st.markdown("**BART (Abstractive)**")
            st.write(bart_sum)
            st.markdown("**TextRank (Extractive)**")
            st.write(tr_sum)
            st.markdown("**Quick Comparison**")
            comp = compare_summaries(text_input, bart_sum, tr_sum)
            st.write(comp)

        if show_keywords:
            st.subheader("🏷️ Top Keywords (TF-IDF)")
            kw = extract_keywords(text_input, top_n=10)
            st.write(", ".join(kw))

        if show_wc:
            st.subheader("☁️ Wordcloud (Summary)")
            generate_wordcloud(summary)

        if show_emotions:
            st.subheader("💡 Emotion Analysis")
            emotions = detect_emotions(text_input, return_all=True)
            st.write("Dominant emotion:", emotions[0][0], f"({emotions[0][1]*100:.2f}% confidence)")
            plot_emotion_bar(emotions)
        st.caption("Models used: facebook/bart-large-cnn (summarization) and bhadresh-savani/distilbert-base-uncased-emotion (emotion detection).")
