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
        # Preprocess input
        user_input_clean = re.sub(r"[^a-zA-Z0-9\s]", "", user_input.lower())

        # Convert text to sequence and pad
        seq = tokenizer.texts_to_sequences([user_input_clean])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        # Predict
        prob_real = float(model.predict(padded)[0][0])
        prob_real = np.clip(prob_real, 0, 1)
        prob_fake = 1 - prob_real

        # Determine label
        label = "Real" if prob_real >= 0.5 else "Fake"

        # Display prediction and probabilities
        st.metric("Prediction", f"{label} ({max(prob_real, prob_fake)*100:.1f}%)")
        st.write(f"Real probability: {prob_real*100:.1f}%")
        st.write(f"Fake probability: {prob_fake*100:.1f}%")

        # Decode tokens from cleaned input
        tokens = tokenizer.texts_to_sequences([user_input_clean])[0]
        index_word = {v: k for k, v in tokenizer.word_index.items()}
        decoded_tokens = [index_word.get(t, "<OOV>") for t in tokens[:20]]

        # Highlight OOV words
        decoded_tokens_highlighted = [
            f"**{w}**" if w == "<OOV>" else w for w in decoded_tokens
        ]
        st.write("Tokens (first 20 words, `<OOV>` highlighted):", decoded_tokens_highlighted)
