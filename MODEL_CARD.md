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

## Error analysis

A real confusion matrix and per-class breakdown on the **full held-out test set (2,000
examples)**, regenerated from the shipped weights with `python -m emotion.error_report`.

![Confusion matrix](assets/confusion_matrix.png)

<details><summary>Confusion matrix as counts (rows = true, cols = predicted)</summary>

| true ↓ / pred → | sadness | joy | love | anger | fear | surprise | recall |
|---|---|---|---|---|---|---|---|
| sadness | 558 | 10 | 2 | 4 | 7 | 0 | 0.96 |
| joy | 6 | 656 | 28 | 3 | 1 | 1 | 0.94 |
| love | 0 | 28 | 128 | 3 | 0 | 0 | 0.81 |
| anger | 13 | 4 | 0 | 246 | 12 | 0 | 0.89 |
| fear | 3 | 0 | 0 | 2 | 208 | 11 | 0.93 |
| surprise | 3 | 7 | 0 | 0 | 12 | 44 | 0.67 |

</details>

**Per-class precision / recall / F1**

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| sadness | 0.957 | 0.960 | 0.959 | 581 |
| joy | 0.930 | 0.944 | 0.937 | 695 |
| love | 0.810 | 0.805 | 0.808 | 159 |
| anger | 0.953 | 0.895 | 0.923 | 275 |
| fear | 0.867 | 0.929 | 0.897 | 224 |
| surprise | 0.786 | 0.667 | 0.721 | 66 |

**Where it fails.** The single largest error axis is **joy ↔ love** (28 + 28 mutual
misclassifications): both are short, affect-positive messages, so the model leans toward the
higher-frequency neighbour. The rarest class, `surprise` (n=66), leaks mainly into `fear` (12)
and `joy` (7). The mistakes are semantically adjacent rather than random — the model learned the
manifold and is mostly losing the low-support classes, not misfiring broadly.

**Confidently wrong (highest-confidence mistakes)** — the cases the model got wrong *and* was
sure about, the slice worth reading:

| true | predicted | conf | text |
|---|---|---|---|
| joy | sadness | 0.99 | i feel very saddened that the king whom i once quite respected as far as monarchs go was i… |
| love | joy | 0.99 | i feel affirmed gracious sensuous and will have less self doubt when a href http generatio… |
| sadness | joy | 0.99 | i first started reading city of dark magic i thought it would be a challenge to actually e… |
| anger | sadness | 0.98 | i actually was in a meeting last week where someone yelled at an older lady because her ph… |
| sadness | joy | 0.98 | i felt a stronger wish to be free from self cherishing through my refuge practice and a re… |
| anger | sadness | 0.98 | i really dont like quinn because i feel like she will just end up hurting barney and i hat… |

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
