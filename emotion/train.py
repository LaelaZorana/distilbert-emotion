"""
Fine-tune DistilBERT on the emotion dataset.

A real fine-tune, not a toy: it loads distilbert-base-uncased, attaches a 6-way
classification head, and trains with a plain PyTorch loop (AdamW + linear warmup) on a
subset of the emotion training set. On a laptop CPU it finishes in a few minutes and
reaches strong accuracy because the messages are short and the classes are distinct. The
fine-tuned model + tokenizer are saved to model/ so the demo and tests run inference
without retraining.

Run:  python -m emotion.train --train-size 6000 --epochs 3
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from emotion.data import ID2LABEL, LABEL2ID, LABELS, MAX_LEN, load_split
from emotion.model import BASE_MODEL, LOCAL_MODEL_DIR


def _encode(texts, tok):
    return tok(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")


def _batches(n, bs):
    for i in range(0, n, bs):
        yield i, min(i + bs, n)


def train(train_size: int = 6000, epochs: int = 3, lr: float = 2e-5,
          batch_size: int = 16, seed: int = 0) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    tr_texts, tr_labels = load_split("train", limit=train_size)
    va_texts, va_labels = load_split("validation")  # full validation set
    tr_labels_t = torch.tensor(tr_labels)

    enc_tr = _encode(tr_texts, tok)
    enc_va = _encode(va_texts, tok)
    va_labels_t = torch.tensor(va_labels)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(tr_texts) + batch_size - 1) // batch_size * epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    def val_accuracy() -> float:
        model.eval()
        preds = []
        with torch.no_grad():
            for a, b in _batches(len(va_texts), 64):
                logits = model(input_ids=enc_va["input_ids"][a:b],
                               attention_mask=enc_va["attention_mask"][a:b]).logits
                preds.append(logits.argmax(dim=1))
        preds = torch.cat(preds)
        return float((preds == va_labels_t).float().mean())

    perm_base = np.arange(len(tr_texts))
    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(perm_base)
        running = 0.0
        step = 0
        n_steps = (len(tr_texts) + batch_size - 1) // batch_size
        for a, b in _batches(len(tr_texts), batch_size):
            idx = perm[a:b]
            optim.zero_grad()
            logits = model(input_ids=enc_tr["input_ids"][idx],
                           attention_mask=enc_tr["attention_mask"][idx]).logits
            loss = loss_fn(logits, tr_labels_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item() * len(idx)
            step += 1
            if step % 50 == 0:
                print(f"  epoch {ep} step {step}/{n_steps}  loss {loss.item():.4f}", flush=True)
        acc = val_accuracy()
        print(f"epoch {ep}  train_loss {running / len(tr_texts):.4f}  val_acc {acc:.4f}", flush=True)
        if acc >= best:
            best = acc
            LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(LOCAL_MODEL_DIR)
            tok.save_pretrained(LOCAL_MODEL_DIR)

    print(f"\nBest val accuracy: {best:.4f}  ->  saved {LOCAL_MODEL_DIR}", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(train_size=args.train_size, epochs=args.epochs, lr=args.lr,
          batch_size=args.batch_size, seed=args.seed)
