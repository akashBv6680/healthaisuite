# app.py
import streamlit as st
import os, sys, json, time, uuid, tempfile, re
from datetime import datetime
import requests

# --- Fix sqlite3 <-> pysqlite3 for Chroma on Streamlit Cloud ---
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
    st.error("pysqlite3 is not installed. Add 'pysqlite3-binary' to requirements.txt.")
    st.stop()

# --- Vector DB + Embeddings ---
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===============================
# Configuration
# ===============================
st.set_page_config(page_title="HealthAI Assistant", page_icon="💙", layout="wide")

COLLECTION_NAME = "healthai_kb"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

LANGUAGE = "English"  # keep single language for clarity; you can expand later

# API key via Streamlit secrets (Settings → Secrets on Streamlit Cloud)
TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY", None)
if not TOGETHER_API_KEY:
    st.warning("No TOGETHER_API_KEY found in st.secrets. Add it before deploying.")

# ===============================
# Embedded Knowledge Base (HealthAI)
# Pulled and condensed from your HealthAI document into atomic passages.
# ===============================
KB_PASSAGES = [
    # Project Title & Domain
    "Project Title: HealthAI Suite — Intelligent Analytics for Patient Care. Domain: Healthcare. Aim: end-to-end AI/ML for EHR, diagnostics, medical text, and feedback to support clinical decisions, engagement, and efficiency.",
    # Problem Statement (bullets collapsed)
    "Problem Statement: Predict outcomes (regression), classify disease risks, discover patient subgroups (clustering), mine associations, compare deep models (NN/CNN/RNN/LSTM), leverage BioBERT/ClinicalBERT, build healthcare chatbot, multilingual translator, and sentiment analysis for patient feedback.",
    # Business Use Cases
    "Business Use Cases: Risk Stratification, Length of Stay Prediction, Patient Segmentation, Association Rules for comorbidities, Imaging Diagnostics via CNN, Time-Series forecasting with RNN/LSTM, Clinical NLP with BioBERT/ClinicalBERT, Healthcare Chatbot for triage/FAQ, Translator for multilingual care, Sentiment for QA.",
    # Approach summary
    "Approach: Data prep (cleaning, imputation, normalization), EDA & feature engineering (BMI, BP, cholesterol, glucose, meds history), modular ML/DL builds per task, evaluation per modality, integration via Streamlit (dashboard) + FastAPI (API).",
    # Modeling choices
    "Modeling: Classification (logistic regression, XGBoost, NN), Regression (linear models, LSTM), Clustering (k-means, HDBSCAN), Association (Apriori), Imaging (CNN on chest X-rays), Time-Series (LSTM on vitals), NLP (BioBERT for notes), Translation (MarianMT), Sentiment (finetuned BERT).",
    # Evaluation
    "Evaluation: Classification (Accuracy, F1, ROC-AUC), Regression (RMSE, MAE, R²), Clustering (Silhouette, Calinski–Harabasz, interpretability), Associations (Support, Confidence, Lift), Imaging (Accuracy, Precision, Recall, AUC), Time-Series (forecast RMSE, early warning detection), NLP (BLEU/COMET, F1 for NER), Sentiment (Precision/Recall, MCC), Chatbot (relevance, faithfulness, latency).",
    # Datasets
    "Datasets: MIMIC-III/IV (EHR), PhysioNet (time-series vitals), NIH Chest X-ray 14, patient feedback datasets (e.g., Kaggle), or synthetic anonymized data.",
    # Data formats and variables
    "Data Formats: CSV/Parquet (tabular), JPG/PNG (images), TXT (notes). Variables: age, gender, vitals, labs, diagnoses, medications, procedures, outcomes.",
    # Preprocessing
    "Preprocessing: Imputation, one-hot encoding for categorical, z-score normalization for vitals, tokenization (BioBERT/ClinicalBERT), image resize/augmentation, chronological train/val/test split.",
    # Deliverables
    "Deliverables: Organized repo, preprocessing scripts, modeling notebooks, trained artifacts, FastAPI endpoints, Streamlit dashboard with patient risk predictions, documentation (README, model cards), final report & slides, demo video.",
    # Guidelines
    "Guidelines: Version control with Git, reproducibility (seeds, configs), data security (PII anonymization, HIPAA/GDPR), experiment tracking (MLflow/W&B), coding standards (PEP8, tests), deployment (containers, REST APIs), ethical AI (fairness, transparency, SHAP/LIME), documentation with pipeline diagrams and user guide.",
    # Timeline
    "Timeline: 10 days overall for build & integration.",
    # Results targets
    "Target Results: >80% F1 in disease classification, improved MAE for LOS, meaningful patient clusters, interpretable association rules, CNN at ≥ baseline clinician accuracy for specific pathologies, LSTM early warnings, better NLP with BioBERT, chatbot grounded answers with low error, translation BLEU > baseline, sentiment detects dissatisfaction trends."
]

# Quick structured “Project Description” (exact fields)
PROJECT_DESCRIPTION = {
    "Title": "HealthAI Suite — Intelligent Analytics for Patient Care",
    "Domain": "Healthcare",
    "Problem Statement": [
        "Predict outcomes (regression)",
        "Classify disease risk categories",
        "Discover patient subgroups (clustering)",
        "Mine medical associations",
        "Compare deep models (NN/CNN/RNN/LSTM)",
        "Use BioBERT/ClinicalBERT for clinical NLP",
        "Healthcare chatbot with RAG",
        "Multilingual translator",
        "Sentiment analysis of feedback",
    ],
    "Business Use Cases": [
        "Risk Stratification",
        "Length of Stay Prediction",
        "Patient Segmentation",
        "Association Rules for comorbidities",
        "Imaging Diagnostics (CNN)",
        "Time-Series Forecasting (RNN/LSTM)",
        "Clinical NLP (BioBERT/ClinicalBERT)",
        "Healthcare Chatbot",
        "Translator",
        "Sentiment Analytics",
    ],
    "Datasets": [
        "MIMIC-III/IV (EHR)",
        "PhysioNet (vitals time-series)",
        "NIH Chest X-ray 14",
        "Patient feedback datasets / synthetic",
    ],
    "Key Variables": ["age", "gender", "vitals", "labs", "diagnoses", "medications", "procedures", "outcomes"],
    "Preprocessing": [
        "Missing value imputation",
        "One-hot categorical encoding",
        "z-score normalization (vitals)",
        "Tokenization (BioBERT/ClinicalBERT)",
        "Image resize & augmentation",
        "Chronological train/val/test split",
    ],
    "Modeling": {
        "Classification": ["Logistic Regression", "XGBoost", "Neural Network"],
        "Regression": ["Linear Models", "LSTM (time-series)"],
        "Clustering": ["k-means", "HDBSCAN"],
        "Association": ["Apriori"],
        "Imaging": ["CNN"],
        "Time Series": ["LSTM"],
        "NLP": ["BioBERT/ClinicalBERT"],
        "Translation": ["MarianMT"],
        "Sentiment": ["Finetuned BERT"],
    },
    "Evaluation": {
        "Classification": ["Accuracy", "F1", "ROC-AUC"],
        "Regression": ["RMSE", "MAE", "R²"],
        "Clustering": ["Silhouette", "Calinski–Harabasz", "Interpretability"],
        "Associations": ["Support", "Confidence", "Lift"],
        "Imaging": ["Accuracy", "Precision", "Recall", "AUC"],
        "Time Series": ["Forecast RMSE", "Early Warning Detection"],
        "NLP": ["BLEU", "COMET", "F1 (NER)"],
        "Sentiment": ["Precision/Recall", "MCC"],
        "Chatbot": ["Relevance", "Faithfulness", "Latency"],
    },
    "Deliverables": [
        "Repo + scripts + notebooks",
        "Trained models",
        "FastAPI endpoints",
        "Streamlit dashboard",
        "Documentation + model cards",
        "Report + slides + demo video",
    ],
    "Guidelines": [
        "Git workflow",
        "Reproducibility (seeds/configs)",
        "HIPAA/GDPR & anonymization",
        "MLflow or W&B",
        "PEP8, tests, modular code",
        "Containers + REST APIs",
        "Ethical AI (fairness/SHAP/LIME)",
        "Diagrams & user guide",
    ],
    "Timeline": "10 days",
    "Target Results": [
        ">80% F1 in disease classification",
        "Improved MAE for LOS",
        "Meaningful clusters",
        "Interpretable association rules",
        "CNN ≥ baseline clinician accuracy",
        "LSTM early warnings",
        "BioBERT boosts clinical NLP",
        "Grounded chatbot with low error",
        "Translator > baseline BLEU",
        "Sentiment flags dissatisfaction trends",
    ],
}

# ===============================
# Caching heavy deps
# ===============================
@st.cache_resource(show_spinner=False)
def init_deps():
    db_path = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=db_path)
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return client, model

@st.cache_resource(show_spinner=False)
def load_kb_into_chroma(client, embedder, collection_name=COLLECTION_NAME):
    col = client.get_or_create_collection(name=collection_name)
    if col.count() == 0:
        # split passages into chunks (defensive, though short already)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = []
        for p in KB_PASSAGES:
            docs.extend(splitter.split_text(p))
        embs = embedder.encode(docs).tolist()
        ids = [str(uuid.uuid4()) for _ in docs]
        col.add(documents=docs, embeddings=embs, ids=ids)
    return col

# ===============================
# Utilities
# ===============================
def search_kb(query, embedder, collection, k=5):
    qemb = embedder.encode([query]).tolist()
    res = collection.query(query_embeddings=qemb[0], n_results=k)
    return (res.get("documents") or [[]])[0]

def call_llm(prompt: str, max_tokens=512, temperature=0.3):
    if not TOGETHER_API_KEY:
        return "⚠️ Missing Together API key. Add TOGETHER_API_KEY to Streamlit secrets."
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {TOGETHER_API_KEY}"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful healthcare project assistant. Keep answers concise and grounded in the provided context."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        r = requests.post(TOGETHER_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"LLM error: {e}"

def rag_answer(question: str, embedder, collection):
    ctx_docs = search_kb(question, embedder, collection, k=6)
    if not ctx_docs:
        return "I don’t have enough context yet."
    context = "\n\n".join(ctx_docs)
    prompt = (
        "Use the following HealthAI knowledge base to answer the question. "
        "If you are not sure, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return call_llm(prompt)

# tiny demo risk predictor (rule-of-thumb)
def demo_predict_risk(age:int=None, bmi:float=None, sbp:int=None, glucose:float=None, smoker:bool=None):
    score = 0
    if age is not None: score += (age >= 60) * 2 + (45 <= age < 60) * 1
    if bmi is not None: score += (bmi >= 30) * 2 + (25 <= bmi < 30) * 1
    if sbp is not None: score += (sbp >= 140) * 2 + (130 <= sbp < 140) * 1
    if glucose is not None: score += (glucose >= 126) * 2 + (110 <= glucose < 126) * 1
    if smoker is not None: score += (1 if smoker else 0)
    label = "Low" if score <= 1 else ("Moderate" if score <= 3 else "High")
    return {"risk_label": label, "score": int(score)}

def handle_special_commands(text: str):
    txt = text.strip().lower()
    if txt in ("/project", "project description", "project", "describe project"):
        # pretty print the fields
        lines = ["**Project Description**"]
        for k, v in PROJECT_DESCRIPTION.items():
            if isinstance(v, dict):
                lines.append(f"- **{k}:**")
                for kk, vv in v.items():
                    lines.append(f"  - **{kk}:** {', '.join(vv)}")
            elif isinstance(v, list):
                lines.append(f"- **{k}:** {', '.join(v)}")
            else:
                lines.append(f"- **{k}:** {v}")
        return "\n".join(lines)

    if txt.startswith("/predict"):
        # /predict age=55 bmi=29 sbp=135 glucose=118 smoker=false
        kv = dict(re.findall(r"(\w+)\s*=\s*([^\s]+)", text))
        def to_bool(v): return str(v).lower() in ("1","true","yes","y")
        age = int(kv["age"]) if "age" in kv else None
        bmi = float(kv["bmi"]) if "bmi" in kv else None
        sbp = int(kv["sbp"]) if "sbp" in kv else None
        glucose = float(kv["glucose"]) if "glucose" in kv else None
        smoker = to_bool(kv["smoker"]) if "smoker" in kv else None
        res = demo_predict_risk(age, bmi, sbp, glucose, smoker)
        return f"**Prediction** → Risk: **{res['risk_label']}** (score={res['score']})."

    return None

# ===============================
# UI State
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

# ===============================
# Load deps & KB
# ===============================
with st.spinner("Starting HealthAI Assistant…"):
    db_client, embedder = init_deps()
    kb_collection = load_kb_into_chroma(db_client, embedder)

# ===============================
# Page body (no uploads, no URL inputs)
# ===============================
st.title("HealthAI Assistant")
st.caption("RAG-powered chatbot with built-in HealthAI knowledge base.")

# Minimal hero / info panel
with st.expander("What can I do?", expanded=False):
    st.markdown(
        """
- Ask about the **HealthAI Suite** (project scope, datasets, metrics, timeline, deliverables).
- Click the chat bubble (bottom-right) to open the chatbot.
- Use **/project** to get exact project fields, or **/predict** with attributes, e.g.:
      """
  )

# ===============================
# Floating Chat Widget (bottom-right)
# ===============================
CHAT_CSS = """
<style>
/* Floating chat button */
#chat-fab {
position: fixed; bottom: 24px; right: 24px; z-index: 1000;
width: 56px; height: 56px; border-radius: 50%; 
background: #1C8CDC; color: white; display: flex; align-items: center; justify-content: center;
box-shadow: 0 6px 20px rgba(0,0,0,0.25); cursor: pointer; user-select: none;
font-size: 24px; font-weight: 700;
}
/* Chat panel */
.chat-panel {
position: fixed; bottom: 92px; right: 24px; z-index: 1000;
width: 380px; max-height: 70vh; background: white; border-radius: 16px;
box-shadow: 0 12px 32px rgba(0,0,0,0.25); overflow: hidden; border: 1px solid #eaeaea;
}
.chat-header {
background: #1C8CDC; color: #fff; padding: 12px 14px; font-weight: 600; display:flex; justify-content:space-between; align-items:center;
}
.chat-body {
padding: 12px; overflow-y: auto; max-height: 54vh;
}
.chat-suggestions {
display: grid; grid-template-columns: 1fr; gap: 8px; margin-top: 6px;
}
.chat-suggestion {
border: 1px solid #eaeaea; border-radius: 10px; padding: 8px 10px; cursor: pointer; background: #fafafa;
}
.chat-footer {
padding: 8px 12px; border-top: 1px solid #efefef; background: #fff;
}
</style>
"""
st.markdown(CHAT_CSS, unsafe_allow_html=True)

# Render FAB
fab_col1, fab_col2, fab_col3 = st.columns([1,1,1])
with fab_col3:
  if st.button("💬", key="fab", help="Chat with HealthAI", use_container_width=False):
      st.session_state.chat_open = not st.session_state.chat_open

# Render Chat Panel
if st.session_state.chat_open:
  # Header
  st.markdown(
      """
<div class="chat-panel">
<div class="chat-header">
  <div>HealthAI Chatbot</div>
  <div>🩺</div>
</div>
<div class="chat-body">
""",
      unsafe_allow_html=True,
  )

  # Initial assistant message & suggestions (only once)
  if not st.session_state.messages:
      greeting = (
          "Hi! I’m your HealthAI assistant. Ask me about the project scope, datasets, metrics, or try **/project** "
          "for exact fields. For a quick demo prediction, use **/predict age=55 bmi=29 sbp=135 glucose=118 smoker=false**.\n\n"
          "_Tap a suggestion below to start:_"
      )
      st.session_state.messages.append({"role": "assistant", "content": greeting})

  # Display chat history
  for m in st.session_state.messages:
      with st.chat_message(m["role"]):
          st.markdown(m["content"])

  # Suggestions block (only when there’s no user message yet)
  if len([m for m in st.session_state.messages if m["role"] == "user"]) == 0:
      cols = st.columns(1)
      with cols[0]:
          st.markdown("**Quick questions**")
          s1 = "What’s the HealthAI Suite about?"
          s2 = "Show the exact project description fields (/project)."
          s3 = "Which datasets and metrics do we use?"
          s4 = "How do you predict health risks? (try /predict)"
          for s in [s1, s2, s3, s4]:
              if st.button(s, key=f"sugg-{s}"):
                  st.session_state.messages.append({"role": "user", "content": s})
                  st.rerun()

  # Footer (chat input)
  user_q = st.chat_input("Ask about HealthAI… (use /project or /predict …)")
  if user_q:
      st.session_state.messages.append({"role": "user", "content": user_q})

      # special commands
      special = handle_special_commands(user_q)
      if special:
          st.session_state.messages.append({"role": "assistant", "content": special})
      else:
          with st.chat_message("assistant"):
              with st.spinner("Thinking…"):
                  reply = rag_answer(user_q, embedder, kb_collection)
                  st.markdown(reply)
          st.session_state.messages.append({"role": "assistant", "content": reply})

      st.rerun()

  st.markdown("</div><div class='chat-footer'>", unsafe_allow_html=True)
  st.caption("Grounded on HealthAI knowledge base • Model: Mistral-7B-Instruct (Together AI)")
  st.markdown("</div></div>", unsafe_allow_html=True)

# ========== End ==========
