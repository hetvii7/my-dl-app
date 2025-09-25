import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

st.set_page_config(page_title="Fake News Detector", page_icon=":mag_right:")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("cnn_fake_news_model.h5")  # ✅ match training filename
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()
MAX_LEN = 50  # ✅ match Colab training

st.title("Fake News Detector — City Flood 2025")
st.write("Enter a news headline about the event and get a Real / Fake prediction.")

user_input = st.text_area("Headline", height=120)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a headline.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        prob = float(model.predict(padded)[0][0])  # ✅ ensure clean float
        label = "Real" if prob >= 0.5 else "Fake"
        st.metric("Prediction", f"{label} ({prob*100:.1f}%)")
        st.progress(min(max(int(prob*100), 0), 100))
        tokens = tokenizer.texts_to_sequences([user_input])[0]

# Convert token IDs back to words
index_word = {v: k for k, v in tokenizer.word_index.items()}
decoded_tokens = [index_word.get(t, "<OOV>") for t in tokens[:20]]

st.write("Tokens (first 20 words):", decoded_tokens)


