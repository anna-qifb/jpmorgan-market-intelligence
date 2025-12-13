import streamlit as st
import requests
from datetime import datetime, timedelta
import pytz
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
import re

st.set_page_config(page_title="JPMorgan Market Intelligence", layout="wide")
st.title("JPMorgan Chase – Deep Learning Market Intelligence")

# Load models
@st.cache_resource
def load_models():
    sentiment = pipeline("text-classification", model="fongkw2025/roberta-sentiment-jpm")
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
    return sentiment, generator

with st.spinner("Loading models..."):
    sentiment_pipe, gen_pipe = load_models()

def get_time_from():
    now_utc = datetime.now(pytz.utc)
    return (now_utc - timedelta(hours=72)).strftime('%Y%m%dT%H%M')

def extract_key_points(articles):
    text = " ".join([a.get('title', '') + " " + a.get('summary', '') for a in articles if a.get('summary')])
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'will', 'have', 'says', 'said', 'been', 'are', 'was', 'which', 'their'}
    common = Counter(w for w in words if w not in stop).most_common(10)
    key_words = [w for w, c in common if c > 1]
    return ", ".join(key_words[:8]) or "No significant themes detected"

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, NVDA, TSLA):").upper().strip()

if st.button("Analyze Market Intelligence"):
    if not ticker:
        st.error("Please enter a valid ticker symbol.")
    else:
        with st.spinner("Fetching real-time news and analyzing..."):
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={get_time_from()}&limit=10&topics=financial_markets,finance&sort=LATEST&apikey=WW21DG3S70HJR18U"
            data = requests.get(url).json()

            if 'feed' not in data or not data['feed']:
                st.warning(f"No recent news found for {ticker}. Recommendation: Hold (insufficient data).")
                st.stop()

            # Filter high-relevance articles
            articles = []
            for a in data['feed']:
                rel = 0.0
                for sd in a.get('ticker_sentiment', []):
                    if sd.get('ticker') == ticker:
                        try:
                            rel = float(sd.get('relevance_score', 0))
                        except:
                            rel = 0.0
                        break
                if rel >= 0.7:
                    articles.append(a)

            articles.sort(key=lambda x: max(float(s.get('relevance_score', 0)) for s in x.get('ticker_sentiment', []) if s.get('ticker') == ticker), reverse=True)
            top_articles = articles[:5]

            if not top_articles:
                st.warning(f"No highly relevant news for {ticker}. Recommendation: Hold.")
                st.stop()

            # Sentiment analysis
            sentiment_labels = []
            for a in top_articles:
                summary = a.get('summary', '')
                if summary:
                    result = sentiment_pipe(summary)[0]
                    sentiment_labels.append(result['label'])
                else:
                    sentiment_labels.append('neutral')

            # Distribution
            dist = Counter(sentiment_labels)
            total = len(sentiment_labels)
            pos = (dist['positive'] / total) * 100
            neu = (dist['neutral'] / total) * 100
            neg = (dist['negative'] / total) * 100

            # Recommendation
            recommendation = "Buy" if pos >= 60 else "Sell" if neg >= 40 else "Hold"

            # === Pie Chart (Safe from NaN) ===
            fig, ax = plt.subplots(figsize=(6, 4))
            sizes = [pos if not np.isnan(pos) else 0, neu if not np.isnan(neu) else 100, neg if not np.isnan(neg) else 0]
            ax.pie(sizes, labels=['Positive', 'Neutral', 'Negative'], autopct='%1.1f%%', colors=['#2ecc71', '#f1c40f', '#e74c3c'])
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

            # Key points
            key_points = extract_key_points(top_articles)

            st.success(f"**Final Recommendation: {recommendation}**")

            # Generate report
            prompt = f"""Write a professional investment recommendation report for {ticker}.

Structure:

1. Summary of recent news including routine news and sentiment
2. Sentiment Score distribution and pie chart description
   - Positive: {pos:.1f}%, Neutral: {neu:.1f}%, Negative: {neg:.1f}%
3. Final recommendation
   - Recommendation: {recommendation}
4. Key points extracted from recent news
   - {key_points}

Use formal financial advisor tone."""

            with st.spinner("Generating detailed report..."):
                output = gen_pipe(prompt, max_length=400, temperature=0.7, do_sample=True, truncation=True)[0]['generated_text']

            st.markdown("### Investment Recommendation Report")
            st.write(output)

            # Show raw articles (optional)
            with st.expander("View Raw News Articles"):
                for i, a in enumerate(top_articles, 1):
                    st.write(f"**Article {i}:** {a.get('title', 'No title')}")
                    st.write(a.get('summary', 'No summary'))
                    st.write(f"URL: {a.get('url', 'N/A')}\n")