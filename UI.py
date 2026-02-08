import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open("emotion_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Emotion Assessment System",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Emotion Assessment System")
st.markdown("Analyze emotions from text using **ML (TF-IDF + Logistic Regression)**")

# Emotion → Emoji mapping
emotion_emoji = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
    "love": "❤️",
    "neutral": "😐"
}

# 🔹 Label mapping (FIX)
label_map = {
    0: "joy",
    1: "sadness",
    2: "anger",
    3: "fear",
    4: "surprise",
    5: "love",
    6: "neutral"
}

text = st.text_area(
    "✍️ Enter your text",
    placeholder="I feel really excited about my new job!"
)

if st.button("🔍 Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed_text = tfidf.transform([text])

        prediction = model.predict(transformed_text)[0]
        probabilities = model.predict_proba(transformed_text)[0]

        # 🔹 Convert numeric prediction to emotion
        emotion = label_map[prediction]

        confidence = np.max(probabilities) * 100
        emoji = emotion_emoji.get(emotion, "🤔")

        st.markdown("---")
        st.subheader("🧾 Prediction Result")

        st.success(f"**Emotion:** {emotion.capitalize()} {emoji}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        st.subheader("📊 Emotion Confidence Breakdown")

 for emo, prob in zip(model.classes_, probabilities):
    emo_name = label_map[emo]
    st.progress(float(prob), text=f"{emo_name.capitalize()} — {prob*100:.2f}%")

