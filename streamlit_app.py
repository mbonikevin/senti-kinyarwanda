"""web page where anyone can type a kinyarwanda tweet and see the sentiment"""

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import clean_text, ID_TO_LABEL

MODEL_ID = "mbonikevin/afroxlmr-kinyarwanda-sentiment"
MAX_LEN = 128

# the last two are known weak spots barely any text and a rhetorical question
EXAMPLES = ["Murakoze cyane, ibi ni byiza cyane!",
            "Ibi ni bibi cyane, ntabwo nshimishijwe na gato.",
            "Inama izabera i Kigali ku wa gatatu.",
            "@user @user @user Ntacyo.",
            "Ese koko ni ukuri ibyo bavuga?"]

@st.cache_resource
def load_model():
    """loaded once and kept the hub copy is the model the notebook trained"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # low_cpu_mem_usage keeps a second copy of the weights out of ram on load
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, low_cpu_mem_usage=True)
    model.eval()
    return tokenizer, model

def classify(text):
    """cleans the text exactly as training did then hands back class chances"""
    tokenizer, model = load_model()
    inputs = tokenizer(clean_text(text), truncation=True, max_length=MAX_LEN,
                       return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    return {ID_TO_LABEL[i]: float(probs[i]) for i in range(len(ID_TO_LABEL))}

st.set_page_config(page_title="Kinyarwanda Tweet Sentiment", page_icon="🇷🇼")
st.title("Kinyarwanda Tweet Sentiment")
st.caption("AfroXLMR fine-tuned on AfriSenti Kinyarwanda. Shows a score for each class.")

if "text" not in st.session_state:
    st.session_state.text = EXAMPLES[0]

st.write("**Examples**")
for i, col in enumerate(st.columns(len(EXAMPLES))):
    # a short label keeps five buttons on one row on a phone
    if col.button(f"{i + 1}", help=EXAMPLES[i], use_container_width=True):
        st.session_state.text = EXAMPLES[i]

text = st.text_area("Kinyarwanda tweet", key="text", height=100,
                    placeholder="Andika hano...")

if text and text.strip():
    scores = classify(text)
    winner = max(scores, key=scores.get)
    st.subheader(f"{winner}  ·  {scores[winner]:.0%}")
    for label, score in scores.items():
        st.progress(score, text=f"{label}  {score:.1%}")
else:
    st.info("Type a tweet or pick an example above.")

st.divider()
st.caption("Neutral is the weakest class and neutral against negative is 53% of all "
           "errors. Very short text and rhetorical questions are the known weak spots "
           "— examples 4 and 5.")
