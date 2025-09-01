# app.py
# HealthAI Suite - Multi-use Streamlit app (Inference-focused)
# Pages: Risk Stratification, Length of Stay, Segmentation, Associations, Imaging, Sequence Forecasting, Clinical NLP, Translator, Sentiment, Chat Assistant
# NOTE: This app demonstrates inference flows and uses pretrained models for demo. Replace models with your fine-tuned artifacts for production.

import streamlit as st
import os
import tempfile
import uuid
import numpy as np
import pandas as pd
import base64
from typing import List, Dict, Any
from io import BytesIO

# ML & HF
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    MarianMTModel,
    MarianTokenizer
)
import torchvision.transforms as T
from PIL import Image

# sklearn / clustering / regression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# For association rules (demo)
try:
    from mlxtend.frequent_patterns import apriori, association_rules
except Exception:
    apriori = None
    association_rules = None

# For sequence demo
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="HealthAI Suite", page_icon="🩺", layout="wide")

# Languages (L6 subset you previously provided + some)
LANGUAGE_DICT = {
    "English": "en",
    "Spanish": "es",
    "Arabic": "ar",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Tamil": "ta",
    "Bengali": "bn",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Chinese (Simplified)": "zh-Hans",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr"
}

# Paths for optional custom models (you can put fine-tuned models here)
CUSTOM_TEXT_CLS_MODEL = "./models/text_classifier"   # optional; HF-style model dir
CUSTOM_RESNET_MODEL = "./models/resnet_imaging"      # optional; torchscript or state_dict
CUSTOM_LOS_MODEL = "./models/los_regressor"          # optional

# -------------------------
# Helpers: safe model loaders (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_text_classifier(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """Load a light sequence classification model for demo. Replace with ClinicalBERT for production."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load text classifier model: {e}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_translation_model(src_lang="en", tgt_lang="hi"):
    """Load MarianMT translation model for src->tgt. Returns tokenizer & model."""
    pair = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    try:
        tkn = MarianTokenizer.from_pretrained(pair)
        m = MarianMTModel.from_pretrained(pair)
        m.eval()
        return tkn, m
    except Exception:
        # fallback: return None
        return None, None

@st.cache_resource(show_spinner=False)
def load_sentiment_model(model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        m = AutoModelForSequenceClassification.from_pretrained(model_name)
        m.eval()
        return tok, m
    except Exception:
        return None, None

@st.cache_resource(show_spinner=False)
def load_imaging_model():
    try:
        import torchvision.models as models
        m = models.resnet18(pretrained=True)
        m.eval()
        preprocess = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        return m, preprocess
    except Exception:
        return None, None

# Light default scikit-learn models for demo (trained on synthetic or minimal)
@st.cache_resource(show_spinner=False)
def load_demo_tabular_models():
    # Random forest demo classifier and regressor pipelines (they are untrained: we'll train on synthetic if needed)
    clf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=50, random_state=42))])
    reg = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(n_estimators=50, random_state=42))])
    return clf, reg

# Demo LSTM model: small TF model for sequence forecasting
@st.cache_resource(show_spinner=False)
def load_sequence_model():
    # small LSTM for demo; in practice, replace with your trained model
    model = Sequential([
        LSTM(32, input_shape=(10,1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------
# Utility functions
# -------------------------
def text_classify(text: str, tokenizer, model, labels=None):
    if tokenizer is None or model is None:
        return {"label": "unknown", "score": 0.0}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        if labels:
            lbl = labels[pred]
        else:
            lbl = str(pred)
        return {"label": lbl, "score": float(probs[pred])}

def translate_text(text: str, src: str, tgt: str):
    # Try to load a Marian model for this pair. If unavailable, return original text.
    tkn, m = load_translation_model(src, tgt)
    if tkn is None or m is None:
        return text
    inputs = tkn.prepare_seq2seq_batch([text], return_tensors="pt")
    with torch.no_grad():
        translated = m.generate(**{k: v for k, v in inputs.items()})
    out = tkn.batch_decode(translated, skip_special_tokens=True)[0]
    return out

def sentiment_text(text: str, tokenizer, model):
    if tokenizer is None or model is None:
        return {"label": "unknown", "score": 0.0}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        return {"label": pred, "score": float(probs[pred])}

def preprocess_structured_input(data: Dict[str, Any]):
    # Convert structured data to vector for clustering/regression/classification
    # Choose numeric fields and normalize using simple scaling for the demo
    numeric_keys = ["age", "bmi", "sbp", "dbp", "glucose", "cholesterol"]
    vals = []
    for k in numeric_keys:
        v = data.get(k, 0.0)
        try:
            vals.append(float(v))
        except Exception:
            vals.append(0.0)
    return np.array(vals).reshape(1, -1)

# Association rules demo using categorical columns in a DataFrame
def mine_association_rules(df: pd.DataFrame, cat_cols: List[str], min_support=0.05, min_threshold=1.0):
    if apriori is None or association_rules is None:
        return [{"rule": "mlxtend not available", "support": 0.0, "confidence": 0.0, "lift": 0.0}]
    # One-hot encode categorical features into transactional boolean columns
    tx = pd.get_dummies(df[cat_cols].astype(str))
    frequent = apriori(tx, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return []
    rules = association_rules(frequent, metric="lift", min_threshold=min_threshold)
    out = []
    for _, r in rules.iterrows():
        out.append({"rule": f"{list(r['antecedents'])} => {list(r['consequents'])}",
                    "support": float(r["support"]),
                    "confidence": float(r["confidence"]),
                    "lift": float(r["lift"])})
    return out

# Imaging inference utility
def predict_image(img: Image.Image, model, preprocess, topk=3):
    if model is None or preprocess is None:
        return []
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        topk_idx = probs.argsort()[-topk:][::-1]
        return [(int(i), float(probs[i])) for i in topk_idx]

# -------------------------
# App UI: Sidebar + Navigation
# -------------------------
st.sidebar.title("HealthAI Suite")
menu = st.sidebar.radio("Select Module", [
    "🧑‍⚕️ Risk Stratification",
    "⏱ Length of Stay Prediction",
    "👥 Patient Segmentation",
    "🔗 Medical Associations",
    "🩻 Imaging Diagnostics",
    "📈 Sequence Forecasting",
    "📝 Clinical NLP",
    "🌐 Translator",
    "💬 Sentiment Analysis",
    "💡 Chat Assistant"
])

# Load models (cached loaders)
text_tok, text_model = load_text_classifier()
sent_tok, sent_model = load_sentiment_model()
img_model, img_preproc = load_imaging_model()
demo_clf, demo_reg = load_demo_tabular_models()
seq_model = load_sequence_model()

# -------------------------
# Common patient form fields (used across pages)
# -------------------------
def patient_input_form(key_prefix="p"):
    with st.form(key=f"form_{key_prefix}"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=45, key=f"{key_prefix}_age")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key=f"{key_prefix}_gender")
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, key=f"{key_prefix}_bmi")
            sbp = st.number_input("Systolic BP", min_value=60, max_value=250, value=120, key=f"{key_prefix}_sbp")
            dbp = st.number_input("Diastolic BP", min_value=40, max_value=160, value=80, key=f"{key_prefix}_dbp")
        with col2:
            glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=100, key=f"{key_prefix}_glucose")
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=500, value=180, key=f"{key_prefix}_cholesterol")
            smoker = st.selectbox("Smoker", ["No", "Yes"], index=0, key=f"{key_prefix}_smoker")
            notes = st.text_area("Clinical Notes (optional)", height=120, key=f"{key_prefix}_notes")
        submitted = st.form_submit_button("Run")
    data = {
        "age": int(age),
        "gender": gender,
        "bmi": float(bmi),
        "sbp": float(sbp),
        "dbp": float(dbp),
        "glucose": float(glucose),
        "cholesterol": float(cholesterol),
        "smoker": smoker == "Yes",
        "notes": notes
    }
    return submitted, data

# -------------------------
# Module: Risk Stratification
# -------------------------
if menu == "🧑‍⚕️ Risk Stratification":
    st.title("Risk Stratification")
    st.write("Predict patient risk (Low / Moderate / High). This uses clinical notes + structured features for inference.")

    submitted, pdata = patient_input_form("risk")
    if submitted:
        # Create an input text from structured data + notes
        concat_text = (
            f"Age: {pdata['age']}; Gender: {pdata['gender']}; BMI: {pdata['bmi']}; "
            f"SBP/DBP: {pdata['sbp']}/{pdata['dbp']}; Glucose: {pdata['glucose']}; "
            f"Cholesterol: {pdata['cholesterol']}. Notes: {pdata['notes']}"
        )

        # Primary path: transformer text classifier if available
        if text_tok is not None and text_model is not None:
            labels = ["Low Risk", "Moderate Risk", "High Risk"]  # map to 3-class if fine-tuned
            res = text_classify(concat_text, text_tok, text_model, labels=labels)
            st.success(f"Predicted Risk: **{res['label']}**  (confidence {res['score']:.2f})")
        else:
            # Fallback: simple rule-based scoring
            score = 0
            score += (pdata['age'] >= 60) * 2 + (45 <= pdata['age'] < 60) * 1
            score += (pdata['bmi'] >= 30) * 2 + (25 <= pdata['bmi'] < 30) * 1
            score += (pdata['sbp'] >= 140) * 2 + (130 <= pdata['sbp'] < 140) * 1
            score += (pdata['glucose'] >= 126) * 2 + (110 <= pdata['glucose'] < 126) * 1
            score += (1 if pdata['smoker'] else 0)
            label = "Low" if score <= 1 else ("Moderate" if score <= 3 else "High")
            st.success(f"Predicted Risk: **{label}** (score={score})")

# -------------------------
# Module: Length of Stay Prediction (Regression)
# -------------------------
elif menu == "⏱ Length of Stay Prediction":
    st.title("Length of Stay Prediction")
    st.write("Predict expected hospital length of stay (days). In production, replace the demo regressor with a trained model.")
    submitted, pdata = patient_input_form("los")
    if submitted:
        vec = preprocess_structured_input(pdata)
        # demo: use simple formula + small regressor fallback
        los_est = 3.0 + (pdata['age']/30.0) + (pdata['bmi']/40.0) + (pdata['glucose']/200.0)
        los_est = round(float(los_est), 2)
        st.success(f"Predicted length of stay: **{los_est} days**")
        st.info("This is a demo estimate. For production, load your trained regression model and run inference here.")

# -------------------------
# Module: Patient Segmentation (Clustering)
# -------------------------
elif menu == "👥 Patient Segmentation":
    st.title("Patient Segmentation (Clustering)")
    st.write("Assign patient to a cohort using clustering of structured features.")
    submitted, pdata = patient_input_form("seg")
    if submitted:
        X = preprocess_structured_input(pdata)
        # For demo, fit kmeans on synthetic data if not available
        # Generate synthetic dataset for centroids (only if in-memory)
        rng = np.random.RandomState(42)
        synthetic = rng.normal(loc=[50,25,120,80,100,180], scale=[15,5,20,10,30,40], size=(200,6))
        X_all = np.vstack([synthetic, X])
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_all)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(Xs)
        pred = kmeans.predict(Xs[-1].reshape(1,-1))[0]
        st.success(f"Assigned Cohort: **Cohort {pred+1}**")
        st.info("Cohort labels are unsupervised; interpret with feature distributions and domain expertise.")

# -------------------------
# Module: Medical Associations (Association Rules)
# -------------------------
elif menu == "🔗 Medical Associations":
    st.title("Medical Associations (Association Rule Mining)")
    st.write("Upload a CSV containing categorical patient fields (e.g., BMI_Category, Hypertension, Diabetes) to mine rules.")
    uploaded = st.file_uploader("Upload CSV (required)", type=["csv"])
    min_support = st.slider("Min support", 0.01, 0.5, 0.05)
    min_lift = st.slider("Min lift", 1.0, 5.0, 1.5)
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        # Let user choose categorical columns
        cat_cols = st.multiselect("Select columns to use for association mining", options=list(df.columns))
        if st.button("Mine rules"):
            with st.spinner("Mining association rules..."):
                rules = mine_association_rules(df, cat_cols, min_support=min_support, min_threshold=min_lift)
                if not rules:
                    st.warning("No rules found with given thresholds.")
                else:
                    for r in rules[:50]:
                        st.write(f"Rule: {r['rule']}  — support {r['support']:.3f}, conf {r['confidence']:.3f}, lift {r['lift']:.3f}")

# -------------------------
# Module: Imaging Diagnostics (CNN)
# -------------------------
elif menu == "🩻 Imaging Diagnostics":
    st.title("Imaging Diagnostics (Upload X-ray/CT/MRI)")
    st.write("Upload a medical image; model will produce a demo classification. Replace with your fine-tuned model for real diagnostic use.")
    uploaded_img = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)
        if st.button("Run imaging model"):
            if img_model is not None and img_preproc is not None:
                preds = predict_image(img, img_model, img_preproc, topk=3)
                # For demo we map output indices to generic labels
                labels = ["Class A", "Class B", "Class C", "Class D", "Class E"]
                st.success("Top predictions:")
                for idx, prob in preds:
                    lbl = labels[idx % len(labels)]
                    st.write(f"{lbl} — {prob:.2f}")
            else:
                st.error("Imaging model not available in this environment.")

# -------------------------
# Module: Sequence Forecasting (RNN/LSTM)
# -------------------------
elif menu == "📈 Sequence Forecasting":
    st.title("Sequence Forecasting (Vitals over time)")
    st.write("Demo: paste a short comma-separated sequence of heart rate or oxygen values and forecast next value.")
    seq_input = st.text_area("Enter time series (comma-separated, up to 50 values)", "72,73,75,80,78,76,77")
    window = st.slider("Window length for model", 3, 20, 10)
    if st.button("Forecast next"):
        try:
            vals = [float(x.strip()) for x in seq_input.split(",") if x.strip()!=""]
            if len(vals) < window:
                st.warning("Not enough points for the chosen window.")
            else:
                # For demo, simple last-value persistence or small LSTM predict if saved model available
                # We'll use a naive forecast: mean of last window +/- small noise
                nxt = float(np.mean(vals[-window:]) + np.random.normal(0,0.5))
                st.success(f"Forecasted next value: {nxt:.2f}")
                st.line_chart(vals + [nxt])
        except Exception as e:
            st.error(f"Failed to parse input: {e}")

# -------------------------
# Module: Clinical NLP (BioBERT/ClinicalBERT)
# -------------------------
elif menu == "📝 Clinical NLP":
    st.title("Clinical NLP")
    st.write("Analyze clinical notes (classification/embedding). Recommended: fine-tune ClinicalBERT on labeled clinical data for best results.")
    notes = st.text_area("Paste clinical notes here", height=200)
    if st.button("Analyze notes"):
        if not notes.strip():
            st.warning("Please paste clinical notes to analyze.")
        else:
            # Basic sentiment/label demonstration using the text classifier model
            res = text_classify(notes, text_tok, text_model, labels=None)
            st.write("Classification result (demo):", res)
            st.info("For production use: fine-tune ClinicalBERT/BioBERT to your labels (diagnosis codes, comorbidity tags).")

# -------------------------
# Module: Translator
# -------------------------
elif menu == "🌐 Translator":
    st.title("Translator (Doctor ↔ Patient)")
    st.write("Translate short messages between languages to help bridge communication.")
    src_lang = st.selectbox("Source Language", list(LANGUAGE_DICT.keys()), index=0)
    tgt_lang = st.selectbox("Target Language", list(LANGUAGE_DICT.keys()), index=1)
    text_to_trans = st.text_area("Text to translate", "Please describe your symptoms.")
    if st.button("Translate"):
        src_code = LANGUAGE_DICT.get(src_lang, "en")
        tgt_code = LANGUAGE_DICT.get(tgt_lang, "en")
        # Try Marian; if not available, fallback (return original)
        try:
            # Attempt common Marian pair, else use the generic approach
            tkn, m = load_translation_model(src_code, tgt_code)
            if tkn is None:
                st.warning("Translation model not available for this pair; returning original text.")
                st.write(text_to_trans)
            else:
                inputs = tkn.prepare_seq2seq_batch([text_to_trans], return_tensors="pt")
                with torch.no_grad():
                    translated = m.generate(**{k: v for k, v in inputs.items()})
                out = tkn.batch_decode(translated, skip_special_tokens=True)[0]
                st.success(out)
        except Exception as e:
            st.error(f"Translation failed: {e}")

# -------------------------
# Module: Sentiment Analysis
# -------------------------
elif menu == "💬 Sentiment Analysis":
    st.title("Sentiment Analysis (Patient feedback)")
    txt = st.text_area("Paste patient feedback or reviews", "The nurse was helpful but waiting time was long.")
    if st.button("Analyze sentiment"):
        res = sentiment_text(txt, sent_tok, sent_model)
        st.write("Sentiment analysis (demo):", res)
        # Map model outputs to human-friendly labels if possible
        st.info("Use fine-tuned sentiment models for domain-specific detection (satisfaction, complaint severity).")

# -------------------------
# Module: Chat Assistant
# -------------------------
elif menu == "💡 Chat Assistant":
    st.title("Health Chat Assistant")
    st.write("Ask quick health questions or request the prediction form to be filled.")
    if "assistant_history" not in st.session_state:
        st.session_state["assistant_history"] = []
    user_q = st.chat_input("Ask about symptoms, devices, or request predictions...")
    if user_q:
        # Basic routing: if user asks for prediction, open the relevant page via message
        q_lower = user_q.lower()
        reply = ""
        if "predict" in q_lower or "risk" in q_lower:
            reply = "To run a patient risk prediction, go to the Risk Stratification page and fill in patient details."
        elif "translator" in q_lower or "translate" in q_lower:
            reply = "Use the Translator page to translate between languages for doctor-patient communication."
        elif "imaging" in q_lower:
            reply = "Use Imaging Diagnostics to upload X-rays or CT images for automatic analysis."
        else:
            # fallback: short knowledge-based answers using the simple classifier's outputs or canned replies
            if "blood pressure" in q_lower:
                reply = "Normal blood pressure is <120/80 mmHg. Persistent higher readings suggest hypertension—consult a clinician."
            elif "diabetes" in q_lower:
                reply = "Diabetes symptoms include frequent urination, thirst, unexplained weight loss. Diagnosis is via fasting glucose or HbA1c tests."
            else:
                reply = "I can help with predictions, translations, or image analysis. Ask me to navigate to the right tool."

        st.session_state["assistant_history"].append(("user", user_q))
        st.session_state["assistant_history"].append(("assistant", reply))

    for role, msg in st.session_state["assistant_history"]:
        with st.chat_message(role):
            st.markdown(msg)
