"""Fine-tuned DistilBERT emotion classifier (build + evaluate).

Public surface:
    LABELS, ID2LABEL, LABEL2ID, MAX_LEN, load_split   (emotion.data)
    load_model_and_tokenizer, predict, predict_one     (emotion.model)
"""
from __future__ import annotations

from emotion.data import ID2LABEL, LABEL2ID, LABELS, MAX_LEN, load_split
from emotion.model import (HUB_REPO, load_model_and_tokenizer, predict,
                           predict_one)

__all__ = [
    "LABELS", "ID2LABEL", "LABEL2ID", "MAX_LEN", "load_split",
    "load_model_and_tokenizer", "predict", "predict_one", "HUB_REPO",
]
