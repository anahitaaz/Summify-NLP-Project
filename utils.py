import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
nltk.download('punkt', quiet=True)

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def plot_emotion_bar(emotions):
    labels = [e[0] for e in emotions]
    scores = [e[1] for e in emotions]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(labels, scores)
    ax.set_xlabel("Confidence")
    ax.set_title("Emotion distribution")
    plt.tight_layout()
    st.pyplot(fig)

def text_stats(text):
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    unique = len(set([w.lower() for w in words if w.isalpha()]))
    st.metric("Word Count", len(words))
    st.metric("Sentence Count", len(sentences))
    st.metric("Unique Words", unique)
    avg = round(len(words)/len(sentences),2) if len(sentences)>0 else 0
    st.metric("Avg Words / Sentence", avg)

def extract_keywords(text, top_n=10):
    vect = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vect.fit_transform([text])
    scores = X.toarray().sum(axis=0)
    terms = vect.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return [t for t,s in ranked[:top_n]]
