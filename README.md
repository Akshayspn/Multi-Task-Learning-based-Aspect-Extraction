# Multi-Task Learning for Aspect-Based Sentiment Analysis (ABSA)

## Overview
This repository implements a **Multi-Task Learning** approach for **Aspect-Based Sentiment Analysis (ABSA)** using **BERT**. The model jointly learns **Token Classification** and **Masked Language Modeling (MLM)** to improve aspect identification and classification. Additionally, it includes **comparison models** such as:
- **BERT-BIO Tagger**
- **Standalone MLM**
- **BiLSTM-CRF**
- **GPT2-DistilBERT Fusion Model** (Ensemble Learning)

The system is optimized for performance using **GPU acceleration (CUDA), cuDF (NVIDIA RAPIDS), and spaCy for dependency parsing**.

## Features
- **Multi-task learning** for improved aspect extraction
- **BIO tagging** for explicit and implicit aspect detection
- **Attention visualization** for interpretability
- **Token classification with BERT** for aspect extraction
- **BiLSTM-CRF implementation** for comparison
- **GPT2-DistilBERT Fusion Model for sentiment classification**
- **Evaluation metrics:** F1-score, precision, recall, accuracy, aspect coverage

## Dependencies
Ensure the following libraries are installed before running the code:
```bash
pip install datasets sympy==1.12.0 spacy cudf torch transformers
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12
python -m spacy download en_core_web_sm
```

## Dataset
The model is tested on **Amazon Gift Cards reviews dataset** (`Gift_Cards.jsonl`). It preprocesses the dataset using **spaCy-based dependency parsing** to generate dynamic BIO tags.

### Data Download
Download and unzip the dataset using the following commands:
```bash
!wget "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz"
!gunzip "/content/Gift_Cards.jsonl.gz"
```

## Model Architecture
### 1. Multi-Task Model
- **Shared Encoder:** BERT
- **Token Classification Head:** Identifies explicit/implicit aspects using BIO tagging
- **MLM Head:** Enhances representation learning

### 2. Comparison Models
- **BERT-BIO Tagger:** Uses BERT for sequence labeling
- **Standalone MLM:** Uses BERT for masked language modeling
- **BiLSTM-CRF:** Uses LSTM-based sequence labeling
- **GPT2-DistilBERT Fusion Model:** Uses ensemble learning for sentiment classification

## Data Preprocessing
- **Tokenization:** Uses `BertTokenizerFast`
- **BIO Tagging:** Utilizes **spaCy dependency parsing** for dynamic aspect labeling
- **cuDF Acceleration:** Converts DataFrames to RAPIDS format for faster processing
- **Data Splitting:** 75% training, 15% validation, 10% testing

## Training & Evaluation
### Training
The model is trained using:
- **AdamW optimizer** (learning rate: `3e-5`, weight decay: `0.01`)
- **Gradient accumulation** (`2 steps` to reduce memory usage)
- **Cross-entropy loss with class weighting** to handle class imbalance

```python
train_model(multi_task_model, train_loader, val_loader, device, tokenizer)
```

### Evaluation
- **Token-Level Accuracy**
- **Aspect Accuracy (Explicit & Implicit)**
- **Aspect Coverage Metrics**
- **Sentiment Classification Performance (GPT2-DistilBERT Fusion Model)**
```python
evaluate_model(multi_task_model, test_loader, device, tokenizer, task="multi-task test")
```

## Attention Visualization
The code includes **attention heatmap visualization** for interpretability. It extracts short sentences (`<30 words`) for clearer visualization.
```python
visualize_attention_heatmap(multi_task_model, sentence, tokenizer, device)
```

## Execution Steps
1. **Load and preprocess the dataset**
2. **Initialize models (Multi-Task, BERT-BIO, MLM, BiLSTM-CRF, GPT2-DistilBERT Fusion)**
3. **Train models on labeled dataset**
4. **Evaluate models using multiple metrics**
5. **Visualize attention heatmaps** for insights

## Results
- **Multi-Task Model performs best** in explicit & implicit aspect extraction
- **BERT-BIO performs comparably but lacks robustness in implicit aspects**
- **BiLSTM-CRF struggles with complex dependencies**
- **Standalone MLM helps improve representation learning**
- **GPT2-DistilBERT Fusion Model outperforms individual models for sentiment classification**

## Conclusion
This framework provides a robust pipeline for **Aspect-Based Sentiment Analysis** using **Multi-Task Learning** and various **explainability techniques**. It integrates **attention visualization, BERT-based token classification, LSTM-based sequence labeling, and ensemble sentiment classification** to offer a comparative analysis of different ABSA approaches.

## Future Work
- **Integrate Explainability via Semantic Graphs**
- **Expand dataset to other domains (electronics, movies, etc.)**
- **Enhance model robustness with adversarial training**

## Author
**Akshay Chauhan**

