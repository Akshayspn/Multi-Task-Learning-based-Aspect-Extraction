# Multitask Learning Based Aspect Detection

## Project Overview
This project focuses on **Aspect-Based Sentiment Analysis (ABSA)** by identifying both **implicit and explicit aspects** in customer reviews. The goal of this project is to develop a novel **multitask model** for explicit and implicit aspect detection that can outperform baseline models in detecting aspects at a fine-grained level.

### Key Contributions
- **Multitask Learning**: A novel multitask model using **BERT** for joint **token classification** and **masked language modeling (MLM)**.
- **BIO Tagging for Aspect Detection**: Automatic labeling of aspects using **spaCy dependency parsing and POS tagging**.
- **Comparative Analysis**: Performance comparison with **BERT-based token classification, standalone MLM, and BiLSTM-CRF models**.
- **Attention Visualization**: Heatmaps to analyze token-level attention.

---

## Installation & Dependencies
To set up the environment, install the required dependencies:

```bash
pip install datasets spacy transformers torch scikit-learn tqdm seaborn matplotlib
python -m spacy download en_core_web_sm
```

---

## Dataset
We use the **Yelp Polarity dataset**, which is automatically loaded from the `datasets` library.

- **Preprocessing**:
  - Text is dynamically tagged with BIO labels.
  - Labels are derived using **POS tagging and dependency parsing**.
  - The dataset is split into **train (75%)**, **validation (15%)**, and **test (10%)** sets.

---

## Model Architectures
### 1. **Multitask Model (Proposed)**
- **Shared Encoder:** BERT-based.
- **Tasks:**
  - **Token Classification**: Classifies tokens into `B`, `I`, and `O` labels for aspect detection.
  - **Masked Language Modeling (MLM)**: Helps improve contextual understanding.
- **Loss Function:** Combines both task losses for joint training.

### 2. **Baseline Models**
- **BERT for Token Classification** (BIO tagging approach).
- **Standalone Masked Language Model (MLM)**.
- **BiLSTM-CRF**: Uses an LSTM network with a CRF layer for sequence labeling.

---

## Training & Evaluation
### **Training**
- **Optimizer:** AdamW (learning rate: 3e-5).
- **Scheduler:** Linear learning rate decay.
- **Batch Size:** 16.
- **Epochs:** Adjustable (default: 1).
- **Weighted Loss Function:** Applied to balance class distribution.

### **Evaluation Metrics**
Each model is evaluated on the test set using:
- **F1-Score** (weighted)
- **Precision & Recall**
- **Accuracy**

---

## Attention Visualization
The project provides an **attention heatmap visualization** to analyze what the multitask model focuses on when processing a sentence.

### **Example Output**
The heatmap helps in understanding which tokens contribute to aspect detection.

```python
visualize_attention_heatmap(multi_task_model, "<<Your Sample reviews>>", tokenizer, device)
```

---

## Usage
### **Running the Project**
1. **Preprocess Data:** Automatically handled when running the script.
2. **Train Models:** Execute the training pipeline.
3. **Evaluate Models:** Compare results across different architectures.
4. **Visualize Attention:** Generate attention heatmaps.

### **Command to Run the Script**
```bash
python yelp_final_code.py
```

---

## Results & Findings
- **The proposed multitask model outperforms baselines** in implicit and explicit aspect detection.
- **Incorporating MLM enhances performance** by providing better contextual understanding.
- **Attention maps reveal how the model assigns importance** to different tokens in a sentence.

---

## Future Work
- **Hyperparameter tuning** for further performance improvement.
- **Integration with explainability techniques** such as **semantic graphs**.

---

## Conclusion
This project presents a **novel multitask model** that enhances **Aspect-Based Sentiment Analysis** by jointly learning **token classification and masked language modeling**. The approach is validated through extensive **comparative analysis**, demonstrating its superiority in detecting **both implicit and explicit aspects** in text reviews.

---

## Author
- **Researcher:** [Akshay Chauhan]


For further inquiries, feel free to contact **[akshaychauhan.jss@gmail.com]**.

---
