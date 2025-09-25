import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

st.set_page_config(page_title="Fake News Detector", page_icon=":mag_right:")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("cnn_fake_news_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()
MAX_LEN = 50

st.title("Fake News Detector — City Flood 2025")
st.write("Enter a news headline about the event and get a Real / Fake prediction.")

user_input = st.text_area("Headline", height=120)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a headline.")
    else:
        # Clean the input: lowercase + remove punctuation
        user_input_clean = re.sub(r"[^a-zA-Z0-9\s]", "", user_input.lower())

        # Convert text to sequences and pad
        seq = tokenizer.texts_to_sequences([user_input_clean])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        # Predict
        prob = float(model.predict(padded)[0][0])
        prob = np.clip(prob, 0, 1)  # ensures probability is in [0,1]
        label = "Real" if prob >= 0.5 else "Fake"
        st.metric("Prediction", f"{label} ({prob*100:.1f}%)")
        st.progress(min(max(int(prob*100), 0), 100))

        # Token IDs from cleaned input
        tokens = tokenizer.texts_to_sequences([user_input_clean])[0]

        # Convert token IDs back to words
        index_word = {v: k for k, v in tokenizer.word_index.items()}
        decoded_tokens = [index_word.get(t, "<OOV>") for t in tokens[:20]]

        st.write("Tokens (first 20 words):", decoded_tokens)
