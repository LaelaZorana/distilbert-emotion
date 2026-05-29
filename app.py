"""
Gradio demo for the fine-tuned DistilBERT emotion classifier.

The UI is bespoke, not the stock label-bars: each of the six emotions has its own colour
and emoji, the top prediction renders as a hero card that glows in that emotion's colour,
and the full distribution renders as an animated custom bar chart. All of it is drawn by
the Python function as HTML into a single gr.HTML panel, so the design is fully controlled
rather than themed defaults. Inference runs the real package (emotion/).

Run locally:   pip install -r requirements.txt && python app.py
On Hugging Face Spaces this file is the entry point (app_file: app.py).
"""
from __future__ import annotations

import gradio as gr

from emotion import LABELS, load_model_and_tokenizer, predict_one

# Per-emotion identity: colour + emoji + a one-word read. This is what makes the demo
# feel like it is about emotion, not a generic classifier.
EMOTION_META = {
    "joy":      {"color": "#f59e0b", "emoji": "😄", "blurb": "bright and upbeat"},
    "sadness":  {"color": "#3b82f6", "emoji": "😢", "blurb": "low and heavy"},
    "love":     {"color": "#ec4899", "emoji": "🥰", "blurb": "warm and tender"},
    "anger":    {"color": "#ef4444", "emoji": "😠", "blurb": "hot and sharp"},
    "fear":     {"color": "#8b5cf6", "emoji": "😨", "blurb": "tense and uneasy"},
    "surprise": {"color": "#14b8a6", "emoji": "😲", "blurb": "caught off guard"},
}

_MODEL = None
_TOK = None


def _ensure_loaded():
    global _MODEL, _TOK
    if _MODEL is None:
        _MODEL, _TOK = load_model_and_tokenizer()
    return _MODEL, _TOK


EMPTY_STATE = """
<div class="ez-empty">
  <div class="ez-empty-emoji">💬</div>
  <div class="ez-empty-text">Type a sentence above and I'll read the feeling behind it.</div>
  <div class="ez-empty-sub">Six emotions: joy, sadness, love, anger, fear, surprise.</div>
</div>
"""


def _render(dist: dict) -> str:
    """Render the prediction as a hero card + animated distribution chart (HTML)."""
    ordered = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_p = ordered[0]
    meta = EMOTION_META[top_label]
    conf = round(top_p * 100)

    hero = f"""
    <div class="ez-hero" style="--c:{meta['color']}">
      <div class="ez-hero-emoji">{meta['emoji']}</div>
      <div class="ez-hero-body">
        <div class="ez-hero-label">{top_label}</div>
        <div class="ez-hero-sub">{conf}% confident &middot; {meta['blurb']}</div>
      </div>
      <div class="ez-hero-ring" style="--p:{top_p:.4f}">
        <span>{conf}<small>%</small></span>
      </div>
    </div>
    """

    rows = []
    for label, p in ordered:
        m = EMOTION_META[label]
        pct = round(p * 100)
        rows.append(f"""
        <div class="ez-row">
          <div class="ez-row-name"><span class="ez-emoji">{m['emoji']}</span>{label}</div>
          <div class="ez-track">
            <div class="ez-fill" style="width:{p*100:.2f}%;background:{m['color']}"></div>
          </div>
          <div class="ez-pct">{pct}%</div>
        </div>
        """)
    chart = f'<div class="ez-chart">{"".join(rows)}</div>'
    return f'<div class="ez-result">{hero}{chart}</div>'


def classify(text: str) -> str:
    if not text or not text.strip():
        return EMPTY_STATE
    model, tok = _ensure_loaded()
    _, dist = predict_one(text.strip(), model, tok)
    return _render(dist)


EXAMPLES = [
    "i can't stop smiling, today went better than i ever hoped",
    "i feel so let down after everything we had planned fell apart",
    "my hands are shaking, i really don't think i can walk in there",
    "how dare they take credit for the work i did all weekend",
    "i didn't expect a package on my doorstep and now i'm grinning",
    "i just feel really tender and grateful for the people around me",
]

CSS = """
:root {
  --ez-bg1:#faf5ff; --ez-bg2:#eff6ff; --ez-ink:#1e1b2e; --ez-muted:#6b6880;
  --ez-card:#ffffff; --ez-line:rgba(20,16,40,.08);
  --ez-font:'Plus Jakarta Sans','Inter',system-ui,sans-serif;
}
.gradio-container { max-width: 760px !important; background:
  radial-gradient(1200px 500px at 15% -10%, var(--ez-bg1), transparent 60%),
  radial-gradient(1000px 500px at 110% 10%, var(--ez-bg2), transparent 55%) !important; }
.gradio-container, .gradio-container * { font-family: var(--ez-font); }

/* Header */
#ez-head { text-align:center; padding: 18px 8px 4px; }
#ez-head .ez-pill { display:inline-block; background:#1e1b2e; color:#fff; border-radius:999px;
  padding:5px 13px; font-size:.7rem; font-weight:700; letter-spacing:.08em; margin-bottom:14px; }
#ez-head h1 { margin:0; font-size:2.05rem; font-weight:800; letter-spacing:-.02em; color:var(--ez-ink);
  background:linear-gradient(90deg,#f59e0b,#ec4899,#8b5cf6,#3b82f6); -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent; }
#ez-head p { margin:10px auto 0; max-width:560px; color:var(--ez-muted); font-size:1.02rem; line-height:1.55; }

/* Input card */
#ez-input textarea { font-size:1.08rem !important; border-radius:16px !important; border:1px solid var(--ez-line) !important;
  padding:16px 18px !important; box-shadow:0 8px 30px rgba(30,27,46,.06) !important; background:var(--ez-card) !important; }
#ez-input textarea:focus { border-color:#8b5cf6 !important; box-shadow:0 0 0 4px rgba(139,92,246,.14) !important; }
#ez-go { border-radius:14px !important; font-weight:800 !important; font-size:1rem !important;
  background:linear-gradient(135deg,#8b5cf6,#ec4899) !important; border:none !important; color:#fff !important;
  box-shadow:0 10px 26px rgba(139,92,246,.35) !important; transition:transform .12s ease, box-shadow .12s ease !important; }
#ez-go:hover { transform:translateY(-1px); box-shadow:0 14px 32px rgba(139,92,246,.45) !important; }

/* Results panel */
.ez-result { animation: ez-fade .35s ease both; }
@keyframes ez-fade { from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:none} }

.ez-hero { display:flex; align-items:center; gap:18px; padding:22px 24px; border-radius:20px;
  background:var(--ez-card); border:1px solid var(--ez-line); position:relative; overflow:hidden;
  box-shadow:0 18px 44px color-mix(in srgb, var(--c) 22%, transparent); }
.ez-hero::before { content:""; position:absolute; inset:0; opacity:.10;
  background:radial-gradient(420px 160px at 8% 0%, var(--c), transparent 70%); }
.ez-hero-emoji { font-size:3.1rem; line-height:1; filter:drop-shadow(0 6px 12px color-mix(in srgb,var(--c) 40%,transparent)); }
.ez-hero-body { flex:1; }
.ez-hero-label { font-size:1.9rem; font-weight:800; text-transform:capitalize; color:var(--ez-ink); letter-spacing:-.01em; }
.ez-hero-sub { color:var(--ez-muted); font-size:.98rem; margin-top:2px; }
.ez-hero-ring { width:64px; height:64px; border-radius:50%; display:grid; place-items:center; flex-shrink:0;
  background:conic-gradient(var(--c) calc(var(--p)*360deg), color-mix(in srgb,var(--c) 16%, #fff) 0); }
.ez-hero-ring span { width:50px; height:50px; border-radius:50%; background:var(--ez-card); display:grid; place-items:center;
  font-weight:800; color:var(--ez-ink); font-size:1.02rem; }
.ez-hero-ring small { font-size:.62rem; font-weight:700; color:var(--ez-muted); }

.ez-chart { margin-top:14px; padding:18px 22px; border-radius:18px; background:var(--ez-card);
  border:1px solid var(--ez-line); box-shadow:0 10px 30px rgba(30,27,46,.05); }
.ez-row { display:flex; align-items:center; gap:14px; padding:7px 0; }
.ez-row-name { width:118px; text-transform:capitalize; font-weight:700; color:var(--ez-ink); font-size:.95rem;
  display:flex; align-items:center; gap:8px; }
.ez-emoji { font-size:1.05rem; }
.ez-track { flex:1; height:13px; border-radius:999px; background:#f1eef9; overflow:hidden; }
.ez-fill { height:100%; border-radius:999px; transform-origin:left; animation: ez-grow .65s cubic-bezier(.2,.8,.2,1) both; }
@keyframes ez-grow { from{transform:scaleX(0)} to{transform:scaleX(1)} }
.ez-pct { width:42px; text-align:right; font-variant-numeric:tabular-nums; font-weight:700; color:var(--ez-muted); font-size:.9rem; }

/* Empty state */
.ez-empty { text-align:center; padding:42px 20px; border-radius:20px; background:var(--ez-card);
  border:1px dashed var(--ez-line); }
.ez-empty-emoji { font-size:2.6rem; }
.ez-empty-text { margin-top:10px; font-weight:700; color:var(--ez-ink); font-size:1.05rem; }
.ez-empty-sub { margin-top:4px; color:var(--ez-muted); font-size:.92rem; }

/* Footer */
.ez-footer { margin-top:22px; padding-top:16px; border-top:1px solid var(--ez-line);
  text-align:center; font-size:.88rem; color:var(--ez-muted); line-height:1.9; }
.ez-footer a { text-decoration:none; font-weight:700; color:#8b5cf6; }
.ez-meta { text-align:center; color:var(--ez-muted); font-size:.82rem; margin-top:10px; }
"""

FOOTER = """
<div class="ez-footer">
💬 DistilBERT fine-tuned for emotion by <b>Laela Zorana</b><br>
<a href="https://github.com/LaelaZorana/distilbert-emotion">Source on GitHub</a> &middot;
<a href="https://huggingface.co/LaelaZ/distilbert-emotion">Model on the Hub</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/nn-from-scratch">NN From Scratch</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/cnn-gradcam">CNN + Grad-CAM</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/ai-agent-scenario-qc">Scenario QC</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/rlhf-pairwise-rater">RLHF Rater</a> &middot;
<a href="https://huggingface.co/spaces/LaelaZ/scorm-qa-validator">SCORM QA</a>
</div>
"""

theme = gr.themes.Soft(
    primary_hue="violet", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), gr.themes.GoogleFont("Inter"),
          "system-ui", "sans-serif"],
)

with gr.Blocks(title="Emotion Classifier (DistilBERT)", theme=theme, css=CSS) as demo:
    gr.HTML(
        '<div id="ez-head"><span class="ez-pill">DEEP LEARNING · NLP · TRANSFORMERS</span>'
        "<h1>What's the feeling behind it?</h1>"
        "<p>I fine-tuned DistilBERT to read a sentence and name the emotion underneath it. "
        "Type something below and watch it decide.</p></div>"
    )
    with gr.Group():
        inp = gr.Textbox(label="", lines=3, elem_id="ez-input", show_label=False,
                         placeholder="e.g. i can't believe this actually worked, i'm so relieved")
        go = gr.Button("Read the emotion", elem_id="ez-go", variant="primary")
    gr.Examples(EXAMPLES, inputs=inp, label="Try one of these")

    out = gr.HTML(EMPTY_STATE)

    go.click(classify, inputs=inp, outputs=out)
    inp.submit(classify, inputs=inp, outputs=out)

    gr.HTML(FOOTER)
    gr.HTML('<div class="ez-meta">Runs the real package (emotion/) with the fine-tuned weights. '
            'Held-out test: 92.0% accuracy, 0.874 macro F1. Metrics &amp; failure cases: '
            '<code>python -m emotion.evaluate</code>.</div>')


if __name__ == "__main__":
    demo.launch()
