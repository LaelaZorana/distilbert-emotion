---
title: Emotion Classifier (DistilBERT)
emoji: 💬
colorFrom: purple
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Emotion Classifier (fine-tuned DistilBERT)

Type a sentence and this model names the feeling behind it: **sadness, joy, love, anger,
fear, or surprise**, with the full probability distribution.

It is `distilbert-base-uncased` **fine-tuned** on the emotion dataset. The trained weights
come with this Space, and the same model lives on the Hub at
[LaelaZ/distilbert-emotion](https://huggingface.co/LaelaZ/distilbert-emotion) with a model card.

This is the build-and-prove pattern I use across my portfolio: I do not just train the model,
I report its **held-out accuracy and macro/weighted F1** and surface the cases where it is
**confidently wrong** (run `python -m emotion.evaluate`). Macro F1 matters because the emotion
classes are imbalanced, so accuracy alone would hide weakness on the rare classes.

**Source & full docs:** https://github.com/LaelaZorana/distilbert-emotion
