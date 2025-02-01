# README: Multi-Task Learning for Aspect-Based Sentiment Analysis

## Overview

This project implements a **multi-task learning framework** for **aspect-based sentiment analysis (ABSA)** using **BERT**. The framework unifies explicit and implicit aspect identification tasks while comparing different models, including:

- **Multi-Task BERT** (for token classification & masked language modeling)
- **BERT-BIO Tagger**
- **Standalone MLM** (Masked Language Model)
- **BiLSTM-CRF**

Additionally, the code includes an **attention heatmap visualization** to interpret model decisions on short sentences.

## Requirements

Ensure you have the necessary dependencies installed:

```bash
pip install datasets sympy==1.12.0 spacy cudf torch transformers seaborn matplotlib
```

Additionally, download the **spaCy English model**:

```bash
python -m spacy download en_core_web_sm
```

If required, install **NVIDIA cuDF**:

```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12
```

## Model Architectures

### 1. Multi-Task Model

- Shared **BERT encoder** for both tasks
- **BIO tagging head** for aspect identification
- **Masked Language Modeling (MLM) head** for contextual learning

### 2. Baseline Comparison Models

- **BERT-BIO Tagger**: Standard BERT for token classification
- **Standalone MLM**: BERT-based masked language model
- **BiLSTM-CRF**: Classical sequence labeling model using LSTM with CRF

## Data Processing

The dataset is loaded from a **JSONL file** and preprocessed using:

- **Duplicate removal**
- **Tokenization with spaCy**
- **BIO tagging for aspect identification** (Explicit & Implicit)
- **Conversion to cuDF & HuggingFace Dataset**
- **Splitting into train (75%), validation (15%), and test (10%)**

## Training and Evaluation

The training script performs:

1. **Training with AdamW optimizer**
2. **Gradient accumulation** to handle batch sizes effectively
3. **Evaluation on validation & test sets**
4. **Aspect accuracy and coverage metrics**

To train and evaluate all models:

```bash
!wget "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz"
!gunzip "/content/Gift_Cards.jsonl.gz"
python Amazon_Final_Code.py
```

## Attention Visualization

To interpret model decisions, the script generates **attention heatmaps** for short sentences (<30 words):

- Uses **BERT's last attention layer**
- Plots heatmaps with Seaborn

## Expected Output

- **Training loss and accuracy** per epoch
- **Precision, Recall, F1-score** for aspect extraction
- **Token-level accuracy**
- **Explicit & Implicit aspect identification metrics**
- **Attention heatmaps** for selected sentences

## Notes

- Modify `file_path` to point to your dataset.
- Adjust `max_epochs`, `batch_size`, and `learning rate` for performance tuning.

## Author

Akshay Chauhan

