# DA6401 — Assignment 3: Transformer for Neural Machine Translation

<div align="center">


**Implementation of "Attention Is All You Need" from scratch using PyTorch**

 • [Paper](https://arxiv.org/abs/1706.03762) 
 • [Dataset](https://huggingface.co/datasets/bentrevett/multi30k)

</div>

---

## Overview

This project implements the full Transformer architecture from scratch for Neural Machine Translation (NMT), translating **German → English** using the Multi30k dataset. Every component — scaled dot-product attention, multi-head attention, positional encoding, encoder/decoder stacks, Noam scheduler, and label smoothing — is implemented from first principles without using any pre-built Transformer libraries.

---

## Results

| Metric | Value |
|--------|-------|
| **Test BLEU Score** | **39.52** |
| Best Val Loss | 2.64 |
| Best Epoch | 21 |
| Dataset | Multi30k (29k / 1014 / 1000) |

---

## Project Structure

```
da6401_assignment_3/
│
├── model.py            # Core Transformer architecture
├── train.py            # Training pipeline
├── dataset.py          # Data loading & preprocessing
├── lr_scheduler.py     # Noam Learning Rate Scheduler
├── experiments.py      # W&B Experiments (2.1 – 2.5)
├── requirements.txt    # Dependencies
└── README.md
```

---

## Architecture

```
Input (German)
     │
     ▼
┌─────────────────────────────────┐
│         ENCODER STACK           │
│  ┌───────────────────────────┐  │
│  │  Input Embedding          │  │
│  │  + Positional Encoding    │  │  PE(pos,2i)   = sin(pos/10000^(2i/d))
│  └───────────────────────────┘  │  PE(pos,2i+1) = cos(pos/10000^(2i/d))
│  ┌───────────────────────────┐  │
│  │  × N Encoder Layers       │  │
│  │  ┌─────────────────────┐  │  │
│  │  │ Multi-Head Attention │  │  │  Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
│  │  │ Add & Norm           │  │  │
│  │  │ Feed Forward         │  │  │  FFN(x) = max(0, xW₁+b₁)W₂+b₂
│  │  │ Add & Norm           │  │  │
│  │  └─────────────────────┘  │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
     │  (encoder memory)
     ▼
┌─────────────────────────────────┐
│         DECODER STACK           │
│  ┌───────────────────────────┐  │
│  │  Output Embedding         │  │
│  │  + Positional Encoding    │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │  × N Decoder Layers       │  │
│  │  ┌─────────────────────┐  │  │
│  │  │ Masked MHA           │  │  │  ← causal mask prevents future attending
│  │  │ Add & Norm           │  │  │
│  │  │ Cross-Attention      │  │  │  ← attends over encoder memory
│  │  │ Add & Norm           │  │  │
│  │  │ Feed Forward         │  │  │
│  │  │ Add & Norm           │  │  │
│  │  └─────────────────────┘  │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
     │
     ▼
Linear + Softmax → Output (English)
```

---

## Best Hyperparameters

```python
config = {
    "d_model":      256,    # embedding dimension
    "N":            4,      # number of encoder/decoder layers
    "num_heads":    8,      # attention heads
    "d_ff":         1024,   # feed-forward inner dimension
    "dropout":      0.1,    # dropout rate
    "warmup_steps": 4000,   # Noam scheduler warmup
    "batch_size":   128,    # training batch size
    "num_epochs":   30,     # training epochs
    "smoothing":    0.1,    # label smoothing epsilon
}
```

---

## Setup

### Requirements

```bash
# Python 3.10+ required
python3.10 -m venv venv310
source venv310/bin/activate          # Mac/Linux
# venv310\Scripts\activate           # Windows

pip install -r requirements.txt

# Download spaCy language models
python -m spacy download de_core_news_sm   # German tokenizer
python -m spacy download en_core_web_sm   # English tokenizer
```

### Requirements File

```
torch
numpy
matplotlib
scikit-learn
wandb
datasets
spacy
tqdm
nltk
gdown
```

---

## Training

```bash
python train.py
```

**What happens automatically:**
1. Downloads Multi30k dataset from HuggingFace (first run only, cached)
2. Builds vocabulary from training data using spaCy tokenization
3. Trains with Noam LR scheduler + label smoothing
4. Saves best checkpoint as `checkpoint.pt` (based on val loss)
5. Logs all metrics to W&B
6. Reports final BLEU on test set

**Training on Google Colab (recommended):**
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/da6401_assignment_3')

!pip install -r requirements.txt -q
!python -m spacy download de_core_news_sm
!python -m spacy download en_core_web_sm

!python train.py
```

---

## Inference

The `Transformer` class handles everything end-to-end via `infer()`:

```python
from model import Transformer

# Automatically downloads weights from Google Drive
model = Transformer()
model.eval()

# Translate a German sentence
german = "Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen."
english = model.infer(german)
print(english)
# → "a group of men loading cotton onto a truck ."
```

**What `Transformer.__init__()` does automatically:**
- Loads spaCy tokenizers (`de_core_news_sm`, `en_core_web_sm`)
- Downloads checkpoint from Google Drive via `gdown`
- Rebuilds vocabulary from saved checkpoint
- Loads model weights

---

## Checkpoint

Model weights are hosted on Google Drive and loaded automatically.

> **Google Drive File ID:** `YOUR_GDRIVE_FILE_ID_HERE`

To download manually:
```python
import gdown
gdown.download("https://drive.google.com/uc?id=YOUR_FILE_ID", "checkpoint.pt")
```

---

## W&B Experiments

**Report:** https://wandb.ai/ch22b080-iit-madras/da6401-a3

Run individual experiments:
```bash
python experiments.py --exp 2.1   # Noam vs Fixed LR
python experiments.py --exp 2.2   # 1/√dk scaling ablation
python experiments.py --exp 2.3   # Attention head heatmaps
python experiments.py --exp 2.4   # Sinusoidal PE vs Learned
python experiments.py --exp 2.5   # Label smoothing vs Cross-Entropy

python experiments.py --exp all   # Run all (takes ~4hrs on Colab GPU)
```

### Experiment Summary

| # | Experiment | Key Finding |
|---|-----------|-------------|
| 2.1 | Noam vs Fixed LR | Noam achieves lower loss and higher val accuracy after warmup ends (~epoch 5) |
| 2.2 | Scaling Factor 1/√dₖ | Without scaling, Q/K gradient norms are lower — evidence of softmax saturation |
| 2.3 | Attention Head Visualization | Heads specialise — local (H≈0.63), next-token, long-range; Heads 1&6 redundant |
| 2.4 | PE vs Learned Embeddings | Sinusoidal wins by ~1 BLEU; both similar on short Multi30k sentences |
| 2.5 | Label Smoothing | Smoothing lowers confidence (0.56 vs 0.62) but achieves equal BLEU — better calibration |

---

## Implementation Notes

- **No `torch.nn.MultiheadAttention` used** — fully custom implementation
- **Positional encoding** registered as a `buffer` (not a trainable parameter)
- **Post-LayerNorm** used (matches original paper): `LayerNorm(x + Sublayer(x))`
- **Xavier uniform** weight initialisation
- **Gradient clipping** at 1.0 to prevent exploding gradients
- **Adam** optimizer with β₁=0.9, β₂=0.98, ε=1e-9 (as per paper)

---

## References

- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Multi30k Dataset: [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k)
- Assignment Skeleton: [MiRL-IITM/da6401_assignment_3](https://github.com/MiRL-IITM/da6401_assignment_3)

---

## Github Repository 
https://github.com/Fuzail1929/DA6401_Assignment_3.git
---

## Wandb Report 
https://wandb.ai/ch22b080-iit-madras/da6401-a3/reports/DA6401-Assignment-3---VmlldzoxNjg0NTg4Mg?accessToken=78ft5tcri0wwj30z3j5iqebuhut7e96u6e9e6hfvru51uvflqk68z949pqugo2i8

---

<div align="center">
Mohammed Fuzail · CH22B080
</div>