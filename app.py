"""
Gradio demo for the fine-tuned DistilBERT emotion classifier.

Type a sentence and the model predicts one of six emotions (sadness, joy, love, anger,
fear, surprise), with the full probability distribution. Runs the real package code
(emotion/), loading the fine-tuned weights that ship with the Space.

Run locally:   pip install -r requirements.txt && python app.py
On Hugging Face Spaces this file is the entry point (app_file: app.py).
"""
from __future__ import annotations

import gradio as gr

from emotion import LABELS, load_model_and_tokenizer, predict_one

ACCENT = "#7c3aed"  # violet

_MODEL = None
_TOK = None


def _ensure_loaded():
    global _MODEL, _TOK
    if _MODEL is None:
        _MODEL, _TOK = load_model_and_tokenizer()
    return _MODEL, _TOK


def classify(text: str):
    if not text or not text.strip():
        return {}
    model, tok = _ensure_loaded()
    _, dist = predict_one(text.strip(), model, tok)
    return dist


EXAMPLES = [
    "i can't stop smiling, today went better than i ever hoped",
    "i feel so let down after everything we had planned fell apart",
    "my hands are shaking, i really don't think i can walk in there",
    "how dare they take credit for the work i did all weekend",
    "i didn't expect a package on my doorstep and now i'm grinning",
    "i just feel really tender and grateful for the people around me",
]

CSS = """
:root { --accent: %s; }
.gradio-container { max-width: 1080px !important; }
#hero { background: linear-gradient(135deg, var(--accent), #1e1b4b);
        color:#fff; border-radius:18px; padding:26px 30px; margin-bottom:6px; }
#hero h1 { margin:0 0 8px 0; font-size:1.7rem; font-weight:800; letter-spacing:-.01em; }
#hero p { margin:0; opacity:.93; font-size:1.0rem; line-height:1.5; max-width:780px; }
#hero .pill { display:inline-block; background:rgba(255,255,255,.16); border-radius:999px;
        padding:3px 11px; font-size:.74rem; font-weight:700; margin-bottom:12px; letter-spacing:.04em; }
.footer { margin-top:20px; padding-top:14px; border-top:1px solid rgba(128,128,128,.25);
        font-size:.88rem; text-align:center; opacity:.92; }
.footer a { text-decoration:none; font-weight:700; color:var(--accent); }
""" % ACCENT

FOOTER = """
<div class="footer">
💬 DistilBERT fine-tuned for emotion by <b>Laela Zorana</b> &nbsp;·&nbsp;
<a href="https://github.com/LaelaZorana/distilbert-emotion">Source on GitHub</a> &nbsp;·&nbsp;
<a href="https://huggingface.co/LaelaZ/distilbert-emotion">Model on the Hub</a> &nbsp;·&nbsp;
see also:
🧠 <a href="https://huggingface.co/spaces/LaelaZ/nn-from-scratch">NN From Scratch</a> ·
🔥 <a href="https://huggingface.co/spaces/LaelaZ/cnn-gradcam">CNN + Grad-CAM</a> ·
🔍 <a href="https://huggingface.co/spaces/LaelaZ/ai-agent-scenario-qc">Scenario QC</a> ·
⚖️ <a href="https://huggingface.co/spaces/LaelaZ/rlhf-pairwise-rater">RLHF Rater</a> ·
📦 <a href="https://huggingface.co/spaces/LaelaZ/scorm-qa-validator">SCORM QA</a>
</div>
"""

theme = gr.themes.Soft(primary_hue="violet", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

with gr.Blocks(title="DistilBERT Emotion Classifier", theme=theme, css=CSS) as demo:
    gr.HTML(
        '<div id="hero"><span class="pill">DEEP LEARNING · NLP · TRANSFORMERS</span>'
        "<h1>💬 Emotion Classifier (fine-tuned DistilBERT)</h1>"
        "<p>I fine-tuned DistilBERT on the emotion dataset so it reads a sentence and names "
        "the feeling behind it: sadness, joy, love, anger, fear, or surprise. Type something "
        "below and watch the probabilities. The model is on the Hugging Face Hub with a model "
        "card, and I report its held-out F1 plus the cases where it is confidently wrong.</p></div>"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Your sentence", lines=3,
                             placeholder="Type how someone feels...")
            btn = gr.Button("Classify emotion", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp, label="Try an example")
        out = gr.Label(num_top_classes=6, label="Predicted emotion")
    btn.click(classify, inputs=inp, outputs=out)
    inp.submit(classify, inputs=inp, outputs=out)

    gr.HTML(FOOTER)
    gr.Markdown("*Runs the actual package (`emotion/`) loading the fine-tuned weights. "
                "Base model: distilbert-base-uncased, fine-tuned to 6 emotion classes. "
                "Held-out metrics and failure cases: `python -m emotion.evaluate`.*")


if __name__ == "__main__":
    demo.launch()
