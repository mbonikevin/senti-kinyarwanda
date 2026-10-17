"""web page where anyone can type a kinyarwanda tweet and see the sentiment"""

import sys
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import clean_text, ID_TO_LABEL  # noqa e402

MODEL_ID = "mbonikevin/afroxlmr-kinyarwanda-sentiment"
MAX_LEN = 128

# loaded from the hub so the app always runs the model the notebook trained
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

def classify(text):
    """cleans the text exactly as training did then hands back class chances"""
    if not text or not text.strip():
        return {label: 0.0 for label in ID_TO_LABEL.values()}
    inputs = tokenizer(clean_text(text), truncation=True, max_length=MAX_LEN,
                       return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    return {ID_TO_LABEL[i]: float(probs[i]) for i in range(len(ID_TO_LABEL))}

# the last two are known weak spots barely any text and a rhetorical question
EXAMPLES = [["Murakoze cyane, ibi ni byiza cyane!"],
            ["Ibi ni bibi cyane, ntabwo nshimishijwe na gato."],
            ["Inama izabera i Kigali ku wa gatatu."],
            ["@user @user @user Ntacyo."],
            ["Ese koko ni ukuri ibyo bavuga?"]]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="Kinyarwanda tweet", placeholder="Andika hano..."),
    outputs=gr.Label(num_top_classes=3, label="Predicted sentiment"),
    title="Kinyarwanda Tweet Sentiment",
    description="AfroXLMR fine-tuned on AfriSenti Kinyarwanda. Shows a score for each class.",
    examples=EXAMPLES,
    flagging_mode="never")

if __name__ == "__main__":
    demo.launch()
