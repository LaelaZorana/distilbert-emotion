## Error analysis (held-out test set)

Real evaluation of the fine-tuned weights on the **`test` split of dair-ai/emotion — 2,000 examples the model never saw in training**. Fully reproducible: `python -m emotion.error_report`.

| metric | score |
|---|---|
| accuracy | 0.920 |
| macro F1 | 0.874 |
| weighted F1 | 0.920 |

### Per-class

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| sadness | 0.957 | 0.960 | 0.959 | 581 |
| joy | 0.930 | 0.944 | 0.937 | 695 |
| love | 0.810 | 0.805 | 0.808 | 159 |
| anger | 0.953 | 0.895 | 0.923 | 275 |
| fear | 0.867 | 0.929 | 0.897 | 224 |
| surprise | 0.786 | 0.667 | 0.721 | 66 |

### Confusion matrix

![Confusion matrix](assets/confusion_matrix.png)

<details><summary>Raw counts (rows = true, cols = predicted)</summary>

| true ↓ / pred → | sadness | joy | love | anger | fear | surprise | **recall** |
|---|---|---|---|---|---|---|---|
| **sadness** | **558** | 10 | 2 | 4 | 7 | 0 | 0.96 |
| **joy** | 6 | **656** | 28 | 3 | 1 | 1 | 0.94 |
| **love** | 0 | 28 | **128** | 3 | 0 | 0 | 0.81 |
| **anger** | 13 | 4 | 0 | **246** | 12 | 0 | 0.89 |
| **fear** | 3 | 0 | 0 | 2 | **208** | 11 | 0.93 |
| **surprise** | 3 | 7 | 0 | 0 | 12 | **44** | 0.67 |

</details>

### Where it fails

The dominant confusions are **joy → love** (28), **love → joy** (28), **anger → sadness** (13). The single largest error axis is **joy ↔ love** (28 + 28 mutual misclassifications): both are short, affect-positive messages, so the model leans toward the higher-frequency neighbour. The weakest classes are **surprise** (F1 0.72, n=66), **love** (F1 0.81, n=159) — the two **rarest** in the data — which is exactly why macro F1 (0.874) sits below accuracy (0.920): macro F1 weights every class equally and so exposes the rare-class weakness that accuracy hides. The rarest class, `surprise` (n=66), leaks mainly into `fear` (12) and `joy` (7). The mistakes are semantically adjacent rather than random — the model learned the manifold and is mostly losing the low-support classes, not misfiring broadly.

### Confidently wrong (highest-confidence mistakes)

The most useful slice for debugging: cases the model got wrong *and* was sure about.

| true | predicted | conf | text |
|---|---|---|---|
| joy | sadness | 0.99 | i feel very saddened that the king whom i once quite respected as far as monarchs go was i… |
| love | joy | 0.99 | i feel affirmed gracious sensuous and will have less self doubt when a href http generatio… |
| sadness | joy | 0.99 | i first started reading city of dark magic i thought it would be a challenge to actually e… |
| anger | sadness | 0.98 | i actually was in a meeting last week where someone yelled at an older lady because her ph… |
| sadness | joy | 0.98 | i felt a stronger wish to be free from self cherishing through my refuge practice and a re… |
| anger | sadness | 0.98 | i really dont like quinn because i feel like she will just end up hurting barney and i hat… |
| anger | joy | 0.98 | whenever i put myself in others shoes and try to make the person happy |
| sadness | joy | 0.98 | i remain hopeful that the feeling i have is actually excitement a long missed friend |
| sadness | anger | 0.98 | i hate these feelings in my heart i hate that work stressed me out i hate that cornelius w… |
| love | joy | 0.97 | i walked to school he felt the bounce in his step the overjoyed feelings of youth and the … |
