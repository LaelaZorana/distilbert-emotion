---
license: mit
language:
- en
library_name: transformers
pipeline_tag: text-classification
base_model: distilbert-base-uncased
datasets:
- dair-ai/emotion
tags:
- emotion
- text-classification
- distilbert
- sentiment
metrics:
- accuracy
- f1
widget:
- text: "i can't stop smiling, today went better than i ever hoped"
- text: "my hands are shaking, i really don't think i can walk in there"
- text: "how dare they take credit for the work i did all weekend"
model-index:
- name: distilbert-emotion
  results:
  - task:
      type: text-classification
      name: Emotion Classification
    dataset:
      type: dair-ai/emotion
      name: emotion
      config: split
      split: test
    metrics:
    - type: accuracy
      value: 0.920
      name: Accuracy
    - type: f1
      value: 0.874
      name: Macro F1
---

# distilbert-emotion

`distilbert-base-uncased` fine-tuned on the [emotion](https://huggingface.co/datasets/dair-ai/emotion)
dataset to classify a short English sentence into one of six emotions:
**sadness, joy, love, anger, fear, surprise**.

Built by [Laela Zorana](https://github.com/LaelaZorana). Code, tests, and a live demo:
- GitHub: https://github.com/LaelaZorana/distilbert-emotion
- Demo Space: https://huggingface.co/spaces/LaelaZ/distilbert-emotion

## Usage

```python
from transformers import pipeline
clf = pipeline("text-classification", model="LaelaZ/distilbert-emotion", top_k=None)
clf("i can't stop smiling, today went better than i ever hoped")
# -> [{'label': 'joy', 'score': 0.99}, ...]
```

## Evaluation

Evaluated on the held-out `test` split (2,000 examples the model never trained on). Macro F1
is reported alongside accuracy because the classes are imbalanced (joy and sadness dominate,
surprise is rare), so accuracy alone would overstate performance on the rare classes.

<!-- METRICS:START -->
| metric | score |
|---|---|
| accuracy | 0.920 |
| macro F1 | 0.874 |
| weighted F1 | 0.920 |

Per-class F1: sadness 0.96, joy 0.94, anger 0.92, fear 0.90, love 0.81, surprise 0.72. The two
weakest classes are the two rarest (love n=159, surprise n=66), which is why macro F1 (0.874)
sits below accuracy (0.920): macro F1 weights every class equally and exposes the rare-class
weakness that accuracy hides.
<!-- METRICS:END -->

The repository also surfaces the model's **confidently wrong** predictions (the loudest
mistakes), which is where the model's real limits show.

## Training

- Base model: `distilbert-base-uncased`
- Dataset: `dair-ai/emotion` (split config), 5,000-example training subset
- Objective: cross-entropy over 6 classes
- Optimizer: AdamW, lr 2e-5, linear warmup (10%), gradient clipping at 1.0
- Max sequence length: 64, batch size 16, 3 epochs, CPU

## Limitations

The emotion dataset is short, informal English (tweet-style). The model can be confidently
wrong on sarcasm, mixed feelings, or text unlike the training distribution. It predicts
exactly one of six emotions and has no "neutral" or "other" class.

## License

MIT.
