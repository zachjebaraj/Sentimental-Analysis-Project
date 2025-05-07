import streamlit as st
import pandas as pd
from model import train_model, predict_sentiment
from utils import clean_text

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("📊 Sentiment Analysis for Customer Feedback")

# Load model and vectorizer
@st.cache_resource
def load_model():
    return train_model()

model, vectorizer = load_model()
label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}

# User input
feedback = st.text_area("✍️ Enter Customer Feedback:")

if st.button("Analyze Sentiment"):
    if feedback.strip():
        try:
            cleaned = clean_text(feedback)
            sentiment = predict_sentiment(model, vectorizer, cleaned)
            st.success(f"**Predicted Sentiment:** {label_map[sentiment]}")

            # Save to history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                "feedback": feedback,
                "sentiment": label_map[sentiment]
            })
        except Exception as e:
            st.error(f"An error occurred while analyzing sentiment: {e}")
    else:
        st.warning("Please enter some feedback.")

# Dashboard
if "history" in st.session_state and st.session_state.history:
    st.subheader("📈 Sentiment Trends Dashboard")
    df = pd.DataFrame(st.session_state.history)
    st.bar_chart(df["sentiment"].value_counts())
    st.dataframe(df.tail(5).reset_index(drop=True))
