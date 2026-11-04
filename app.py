from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "news_classifier.joblib"

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")
st.title("📰 Fake and Real News Detection")
st.write("Enter a news headline or article text. The trained model will estimate whether its wording resembles the REAL or FAKE examples in its training data.")

if not MODEL_PATH.exists():
    st.warning("Model not found. First run `python train_model.py` in the project folder.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
news_text = st.text_area("Paste news text here", height=180, placeholder="Enter a headline or news article...")

if st.button("Detect News", type="primary"):
    if len(news_text.strip()) < 20:
        st.error("Please enter at least 20 characters.")
    else:
        prediction = model.predict([news_text])[0]
        probabilities = model.predict_proba([news_text])[0]
        classes = model.classes_
        confidence = float(probabilities[list(classes).index(prediction)]) * 100
        if prediction == "REAL":
            st.success(f"Model prediction: {prediction}")
        else:
            st.error(f"Model prediction: {prediction}")
        st.metric("Model confidence", f"{confidence:.1f}%")
        st.caption("Important: This is a text-pattern prediction, not fact-checking. Verify important claims using trusted sources.")
st.divider()
st.caption("BCA Data Science Minor Project | TF-IDF + Logistic Regression + Streamlit")
