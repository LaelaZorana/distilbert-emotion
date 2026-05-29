"""
Dataset loading for the emotion classifier.

Uses the `emotion` dataset (dair-ai/emotion): short English messages each labelled with
one of six emotions. Tweet-length text, so a small max length is plenty. Training uses a
subset so a real fine-tune finishes on a laptop CPU, while validation/test stay full so
the reported metrics are honest.
"""
from __future__ import annotations

from typing import List, Tuple

# Canonical label order for dair-ai/emotion (label ids 0..5).
LABELS: List[str] = ["sadness", "joy", "love", "anger", "fear", "surprise"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

MAX_LEN = 64


def load_split(split: str, limit: int | None = None) -> Tuple[List[str], List[int]]:
    """Return (texts, labels) for a split ('train'|'validation'|'test').

    If `limit` is set, returns a class-stratified-ish head of that many rows (the dataset
    is shuffled with a fixed seed first so the subset is representative, not front-loaded).
    """
    from datasets import load_dataset

    ds = load_dataset("dair-ai/emotion", "split", split=split)
    ds = ds.shuffle(seed=42)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return list(ds["text"]), list(ds["label"])
