"""
Evaluate the fine-tuned emotion classifier on the held-out test set.

The "I verified it" half. It reports overall accuracy, macro and weighted F1, per-class
precision/recall/F1, and surfaces the model's most CONFIDENTLY WRONG predictions. F1 (not
just accuracy) matters here because the emotion classes are imbalanced: 'joy' and
'sadness' dominate while 'surprise' is rare, so a model can look accurate while quietly
failing the rare classes. Macro F1 exposes that.

Run:  python -m emotion.evaluate
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from emotion.data import LABELS, load_split
from emotion.model import load_model_and_tokenizer, predict


def evaluate_test(limit: int | None = None, verbose: bool = True) -> Dict:
    """Return a metrics dict over the test split (accuracy, macro/weighted F1, per class)."""
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    texts, labels = load_split("test", limit=limit)
    model, tok = load_model_and_tokenizer()

    preds: List[int] = []
    confs: List[float] = []
    # Batch through predict() to keep memory flat.
    for a in range(0, len(texts), 64):
        chunk = texts[a:a + 64]
        for dist in predict(chunk, model, tok):
            top = max(dist, key=dist.get)
            preds.append(LABELS.index(top))
            confs.append(dist[top])

    y_true = np.array(labels)
    y_pred = np.array(preds)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0
    )

    # Most confident mistakes first.
    wrong: List[Tuple[str, str, str, float]] = []
    for i in range(len(texts)):
        if y_pred[i] != y_true[i]:
            wrong.append((texts[i], LABELS[y_true[i]], LABELS[y_pred[i]], confs[i]))
    wrong.sort(key=lambda x: x[3], reverse=True)

    result = {
        "n": len(texts),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(support[i])}
                      for i in range(len(LABELS))},
        "confidently_wrong": wrong[:10],
    }

    if verbose:
        print(f"Held-out test set: {result['n']} examples")
        print(f"Accuracy:    {acc:.3f}")
        print(f"Macro F1:    {macro_f1:.3f}")
        print(f"Weighted F1: {weighted_f1:.3f}\n")
        print(f"{'class':9s} {'prec':>6s} {'recall':>7s} {'f1':>6s} {'n':>5s}")
        for lab in LABELS:
            c = result["per_class"][lab]
            print(f"{lab:9s} {c['precision']:6.3f} {c['recall']:7.3f} {c['f1']:6.3f} {c['support']:5d}")
        print(f"\nConfidently wrong (top {min(5, len(wrong))} of {len(wrong)} misses):")
        for text, true, pred, conf in wrong[:5]:
            snippet = (text[:60] + "...") if len(text) > 60 else text
            print(f"  true={true:8s} pred={pred:8s} conf={conf:.2f}  \"{snippet}\"")
    return result


if __name__ == "__main__":
    evaluate_test(verbose=True)
