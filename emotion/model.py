"""
Model loading and inference for the emotion classifier.

The fine-tuned model is loaded from the local `model/` directory if present (that is what
ships in the repo and runs the demo), otherwise from the HF Hub repo. transformers and
torch are imported lazily inside functions so the package imports cleanly for the parts
that do not need them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from emotion.data import ID2LABEL, LABELS, MAX_LEN

LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
HUB_REPO = "LaelaZ/distilbert-emotion"
BASE_MODEL = "distilbert-base-uncased"


def _resolve_source() -> str:
    """Prefer the shipped local weights; fall back to the Hub."""
    if (LOCAL_MODEL_DIR / "config.json").exists():
        return str(LOCAL_MODEL_DIR)
    return HUB_REPO


def load_model_and_tokenizer(source: str | None = None):
    """Load (model, tokenizer) in eval mode from the local dir or the Hub."""
    import torch  # noqa: F401  (ensures torch backend is present)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src = source or _resolve_source()
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src)
    model.eval()
    return model, tok


def predict(texts: List[str], model, tok) -> List[Dict[str, float]]:
    """Classify a batch of texts. Returns one {label: probability} dict per text."""
    import torch

    if isinstance(texts, str):
        texts = [texts]
    enc = tok(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    out = []
    for row in probs:
        out.append({ID2LABEL[i]: float(row[i]) for i in range(len(LABELS))})
    return out


def predict_one(text: str, model, tok) -> Tuple[str, Dict[str, float]]:
    """Classify one text. Returns (top_label, {label: probability})."""
    dist = predict([text], model, tok)[0]
    top = max(dist, key=dist.get)
    return top, dist
