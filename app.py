import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "bert-base-uncased-sentiment-model"  # change if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Correct id2label mapping (your training order)
ID2LABEL = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}


# ==============================
# LOAD MODEL + TOKENIZER
# ==============================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="SentiMotion", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(-45deg,#141E30,#243B55,#1a2a6c,#000);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: #EAEAEA;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    @keyframes gradient {
        0% {background-position:0% 50%;}
        50% {background-position:100% 50%;}
        100% {background-position:0% 50%;}
    }
    .big-title {
        font-size: 50px;
        font-weight: 900;
        text-align: center;
        color: #f9f9f9;
        text-shadow: 0 0 15px #00c3ff, 0 0 25px #00c3ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='big-title'>✨ SentiMotion ✨</div>", unsafe_allow_html=True)
st.write("Enter a tweet and feel the AI vibes…")

user_text = st.text_area("Your Tweet", placeholder="Type something emotional… 🕊️")

if st.button("🔮 Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            user_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_id = int(probs.argmax())
            pred_label = ID2LABEL[pred_id]

        st.success(f"**Predicted Emotion:** {pred_label}")
        st.write("### Confidence")
        for idx, p in enumerate(probs):
            cols = st.columns([2, 6, 2])   # [label, progress bar, percent]
            with cols[0]:
                st.markdown(f"**{ID2LABEL[idx]}**")
            with cols[1]:
                st.progress(float(p))
            with cols[2]:
                st.markdown(f"**{p*100:.1f}%**")


