"""
Tests for the emotion classifier.

Two layers. The fast layer (always runs, no model download) checks the data contract and
the prediction post-processing logic with a tiny stub model, so the wiring is verified in
isolation and offline. The heavy layer (skipped unless the fine-tuned model is present)
runs the real DistilBERT end to end and asserts the shipped checkpoint clears an F1 bar,
so a regressed or corrupted model fails CI rather than the live demo.
"""
from __future__ import annotations

import numpy as np
import pytest

from emotion.data import ID2LABEL, LABEL2ID, LABELS
from emotion import model as model_mod


def test_label_maps_are_consistent():
    assert len(LABELS) == 6
    assert ID2LABEL[0] == "sadness" and ID2LABEL[1] == "joy"
    for i, lab in enumerate(LABELS):
        assert LABEL2ID[lab] == i
        assert ID2LABEL[i] == lab


class _StubLogits:
    def __init__(self, logits):
        self.logits = logits


class _StubModel:
    """Returns fixed logits regardless of input, to test predict() post-processing."""
    def __init__(self, logits_row):
        import torch
        self._row = torch.tensor(logits_row, dtype=torch.float32)

    def eval(self):
        return self

    def __call__(self, **enc):
        import torch
        n = enc["input_ids"].shape[0]
        return _StubLogits(self._row.unsqueeze(0).repeat(n, 1))


class _StubTok:
    """Minimal tokenizer stub: returns batch tensors of the right leading dimension."""
    def __call__(self, texts, **kw):
        import torch
        n = len(texts)
        return {"input_ids": torch.zeros(n, 4, dtype=torch.long),
                "attention_mask": torch.ones(n, 4, dtype=torch.long)}


def test_predict_returns_probability_distribution():
    import torch  # noqa: F401
    # logits favoring 'joy' (index 1)
    logits = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0]
    model, tok = _StubModel(logits), _StubTok()
    dists = model_mod.predict(["anything", "another"], model, tok)
    assert len(dists) == 2
    for d in dists:
        assert set(d.keys()) == set(LABELS)
        assert abs(sum(d.values()) - 1.0) < 1e-5
        assert max(d, key=d.get) == "joy"
        assert all(0.0 <= v <= 1.0 for v in d.values())


def test_predict_one_picks_argmax_label():
    logits = [0.0, 0.0, 0.0, 6.0, 0.0, 0.0]   # 'anger' (index 3)
    model, tok = _StubModel(logits), _StubTok()
    top, dist = model_mod.predict_one("x", model, tok)
    assert top == "anger"
    assert dist[top] == max(dist.values())


def test_predict_accepts_single_string():
    logits = [0.0, 0.0, 0.0, 0.0, 0.0, 7.0]   # 'surprise'
    model, tok = _StubModel(logits), _StubTok()
    out = model_mod.predict("a single string, not a list", model, tok)
    assert isinstance(out, list) and len(out) == 1


# --- Heavy layer: real fine-tuned model, skipped if not present. ---

def test_finetuned_clears_f1_bar():
    """Prove the shipped model works: macro F1 on held-out test must clear a bar."""
    if not (model_mod.LOCAL_MODEL_DIR / "config.json").exists():
        pytest.skip("fine-tuned model not present (run python -m emotion.train)")
    pytest.importorskip("sklearn")
    pytest.importorskip("datasets")

    from emotion.evaluate import evaluate_test

    res = evaluate_test(limit=500, verbose=False)   # subset of test keeps CI quick
    assert res["accuracy"] >= 0.85, f"accuracy regressed: {res['accuracy']:.3f}"
    assert res["macro_f1"] >= 0.70, f"macro F1 regressed: {res['macro_f1']:.3f}"
