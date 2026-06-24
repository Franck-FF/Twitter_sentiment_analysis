import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "FRANCKF89/twitter-sentiment-distilbert"

id2label = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
    3: "Irrelevant"
}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return id2label[predicted_class]

def predict_multiple(texts):
    results = []

    for text in texts:
        sentiment = predict_sentiment(text)
        results.append({
            "Tweet": text,
            "Prediction": sentiment
        })

    return results

st.title("Twitter Sentiment Analysis")
st.write("This app uses a fine-tuned DistilBERT model to classify tweet sentiment.")

st.write("Enter one tweet or paste several tweets, one per line.")

user_text = st.text_area(
    "Enter tweet(s):",
    height=200,
    placeholder="Example:\nI love this game!\nThis update ruined everything.\nNvidia announced a new graphics card."
)

if st.button("Predict Sentiment"):

    if user_text.strip() == "":
        st.warning("Please enter at least one tweet.")

    else:
        tweets = [
            tweet.strip()
            for tweet in user_text.split("\n")
            if tweet.strip()
        ]

        results = predict_multiple(tweets)

        st.subheader("Predictions")

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
