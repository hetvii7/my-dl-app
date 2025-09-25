import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

st.set_page_config(page_title="Fake News Detector", page_icon=":mag_right:")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("cnn_fake_news_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()
MAX_LEN = 60  # must match training MAX_LEN

st.title("Fake News Detector — City Flood 2025")
st.write("Enter a news headline about the event and get a Real / Fake prediction.")

user_input = st.text_area("Headline", height=120)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a headline.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        prob = model.predict(padded)[0][0]
        label = "Real" if prob >= 0.5 else "Fake"
        st.metric("Prediction", f"{label} ({prob:.2f})")
        st.progress(min(max(int(prob*100),0),100))
        # helpful extra: show explanation (tokens)
        tokens = tokenizer.texts_to_sequences([user_input])[0]
        st.write("Tokens (first 20):", tokens[:20])
