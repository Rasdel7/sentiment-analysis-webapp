import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from textblob import TextBlob
import nltk
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords


st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Real-Time Sentiment Analyzer")
st.markdown("Detect emotions and sentiment in any text instantly.")
st.markdown("---")


def analyze_sentiment(text):
    blob       = TextBlob(text)
    polarity   = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        sentiment = "Positive 😊"
        color     = "#2ecc71"
    elif polarity < -0.1:
        sentiment = "Negative 😢"
        color     = "#e74c3c"
    else:
        sentiment = "Neutral 😐"
        color     = "#f39c12"

    return sentiment, polarity, subjectivity, color


def get_word_freq(text, n=10):
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    words = [w for w in words if w not in stop_words]
    return Counter(words).most_common(n)

tab1, tab2, tab3 = st.tabs([
    "📝 Single Text", "📊 Bulk Analysis", "📈 Live Tracker"
])

with tab1:
    st.markdown("### Analyze Any Text")
    user_text = st.text_area(
        "Enter your text here:",
        placeholder="Type anything — a review, tweet, comment...",
        height=150
    )

    if st.button("Analyze Sentiment", type="primary"):
        if user_text.strip():
            sentiment, polarity, subjectivity, color = analyze_sentiment(
                user_text)

            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment",    sentiment)
            col2.metric("Polarity",     f"{polarity:.3f}",
                        help="-1 = very negative, +1 = very positive")
            col3.metric("Subjectivity", f"{subjectivity:.3f}",
                        help="0 = objective, 1 = subjective")

            st.markdown(
                f"<h2 style='text-align:center; color:{color}'>"
                f"{sentiment}</h2>",
                unsafe_allow_html=True
            )

            
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(['Polarity'], [polarity],
                    color=color, height=0.4)
            ax.barh(['Polarity'], [1],
                    color='#ecf0f1', height=0.4)
            ax.set_xlim(-1, 1)
            ax.axvline(x=0, color='black', linewidth=1)
            ax.set_title('Sentiment Polarity Score', fontsize=12)
            ax.set_xlabel('-1 (Negative) → 0 (Neutral) → +1 (Positive)')
            plt.tight_layout()
            st.pyplot(fig)

           
            word_freq = get_word_freq(user_text)
            if word_freq:
                st.markdown("### 🔤 Most Frequent Words")
                words, counts = zip(*word_freq)
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.bar(words, counts, color='#3498db', edgecolor='black')
                ax2.set_title('Word Frequency')
                ax2.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig2)
        else:
            st.warning("Please enter some text first!")


with tab2:
    st.markdown("### Analyze Multiple Texts at Once")
    st.markdown("Enter one text per line:")

    bulk_text = st.text_area(
        "Bulk input:",
        placeholder="I love this product!\nThis is terrible.\nIt was okay I guess.",
        height=200
    )

    if st.button("Analyze All", type="primary"):
        if bulk_text.strip():
            lines = [l.strip() for l in bulk_text.split('\n')
                     if l.strip()]
            results = []
            for line in lines:
                sentiment, polarity, subjectivity, color = analyze_sentiment(
                    line)
                results.append({
                    'Text':         line[:60] + '...' if len(line) > 60
                                    else line,
                    'Sentiment':    sentiment,
                    'Polarity':     round(polarity, 3),
                    'Subjectivity': round(subjectivity, 3)
                })

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

           
            sentiment_counts = df['Sentiment'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = {'Positive 😊': '#2ecc71',
                      'Negative 😢': '#e74c3c',
                      'Neutral 😐':  '#f39c12'}
            bar_colors = [colors.get(s, '#3498db')
                          for s in sentiment_counts.index]
            ax.bar(sentiment_counts.index, sentiment_counts.values,
                   color=bar_colors, edgecolor='black')
            ax.set_title('Sentiment Distribution')
            ax.set_ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Please enter some texts first!")


with tab3:
    st.markdown("### 📈 Track Sentiment of Multiple Topics")
    st.markdown("Compare sentiment across different texts side by side.")

    topics = {}
    col1, col2 = st.columns(2)

    with col1:
        t1 = st.text_input("Topic 1 name:", value="Product A")
        r1 = st.text_area("Topic 1 reviews (one per line):",
                           value="Great product!\nLove it!\nAmazing quality.",
                           height=100)
    with col2:
        t2 = st.text_input("Topic 2 name:", value="Product B")
        r2 = st.text_area("Topic 2 reviews (one per line):",
                           value="Terrible.\nWaste of money.\nNot recommended.",
                           height=100)

    if st.button("Compare Topics", type="primary"):
        for topic, reviews in [(t1, r1), (t2, r2)]:
            lines = [l.strip() for l in reviews.split('\n') if l.strip()]
            scores = [TextBlob(l).sentiment.polarity for l in lines]
            topics[topic] = sum(scores) / len(scores) if scores else 0

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71' if v >= 0 else '#e74c3c'
                  for v in topics.values()]
        bars = ax.bar(topics.keys(), topics.values(),
                      color=colors, edgecolor='black')
        for bar, val in zip(bars, topics.values()):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=12,
                    fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_title('Average Sentiment Score by Topic', fontsize=13)
        ax.set_ylabel('Polarity Score')
        ax.set_ylim(-1, 1)
        plt.tight_layout()
        st.pyplot(fig)

        winner = max(topics, key=topics.get)
        st.success(f"✅ {winner} has more positive sentiment!")

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Powered by TextBlob & Streamlit"
)