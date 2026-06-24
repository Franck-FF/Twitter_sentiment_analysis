import streamlit as st
import torch
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

st.title("Twitter Sentiment Analysis")
st.write("This app uses a fine-tuned DistilBERT model to classify tweet sentiment.")

user_text = st.text_area("Enter a tweet:")

if st.button("Predict Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter a tweet first.")
    else:
        prediction = predict_sentiment(user_text)
        st.success(f"Predicted Sentiment: {prediction}")