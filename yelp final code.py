# -*- coding: utf-8 -*-
"""
Author: Akshay Chauhan
"""
!pip install datasets
!pip install sympy==1.12.0
!pip install spacy 
!pip install cudf
!pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12 #download directly from NVIDIA if default repo do not work
!python -m spacy download en_core_web_sm
import spacy
import time
# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")
import cudf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,  # Import BertTokenizerFast
    BertForTokenClassification,
    BertForMaskedLM,
    AdamW,
    get_scheduler
)
from datasets import Dataset
# Import libraies to select short sentences for Attention visualisation, else generated heatmap is incomprehensible for longer sentences
import matplotlib.pyplot as plt
import seaborn as sns

# Load Tokenizer using BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Multi-Task Model
class MultiTaskModel(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased"):
        super().__init__()
        self.shared_encoder = BertForTokenClassification.from_pretrained(pretrained_model_name)
        self.token_classification_head = nn.Sequential(
            nn.Dropout(0.5),  # Increased dropout for regularization
            nn.Linear(768, 3)  # BIO tagging
        )
        self.mlm_head = BertForMaskedLM.from_pretrained(pretrained_model_name).cls

    def forward(self, input_ids, attention_mask, token_classification_labels=None, mlm_labels=None):
        shared_output = self.shared_encoder.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        token_classification_logits = self.token_classification_head(shared_output.last_hidden_state)
        mlm_logits = self.mlm_head(shared_output.last_hidden_state)

        loss = 0
        if token_classification_labels is not None:
            class_weights = torch.tensor([1.0, 2.0, 2.0]).to(device)  # Class weighting for imbalance
            loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            token_classification_loss = loss_fn(token_classification_logits.view(-1, 3), token_classification_labels.view(-1))
            loss += token_classification_loss

        if mlm_labels is not None:
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(mlm_logits.view(-1, tokenizer.vocab_size), mlm_labels.view(-1))
            loss += mlm_loss

        return loss, token_classification_logits, mlm_logits

# Comparison Models
class BERTBIOTagger(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", num_labels=3):
        super().__init__()
        self.model = BertForTokenClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss if labels is not None else None
        logits = outputs.logits
        return loss, logits

class StandaloneMLM(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased"):
        super().__init__()
        self.model = BertForMaskedLM.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask, labels=labels)
        loss = outputs.loss if labels is not None else None
        logits = outputs.logits
        return loss, logits

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512, output_dim=3):  # Increased dimensions
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits


# Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    reviews_df = pd.read_json(file_path, lines=True) # Load with pandas first
    reviews_df = reviews_df.drop_duplicates(subset="text").reset_index(drop=True)
        # Dynamic BIO Tagging Logic
    def label_bio_tags_dynamic_v2(text):
      doc = nlp(text)
      tokens = [token.text for token in doc]
      tags = ["O"] * len(tokens)

      for i, token in enumerate(doc):
          # Explicit aspect (nouns, adjectives)
          if token.pos_ in {"NOUN", "ADJ"} and token.dep_ in {"amod", "nsubj", "dobj"}:
              tags[i] = "B-explicit" if tags[i] == "O" else "I-explicit"

          # Implicit aspect (verbs, auxiliary with negation)
          if token.pos_ in {"AUX", "VERB"} and token.dep_ == "ROOT" and "neg" in [child.dep_ for child in token.children]:
              for child in token.children:
                  if child.dep_ in {"attr", "acomp", "xcomp", "pobj"}:
                      child_index = tokens.index(child.text)
                      tags[child_index] = "B-implicit" if tags[child_index] == "O" else "I-implicit"

      # The label mapping should only have three classes for BIO tagging:
      label_mapping = {"O": 0, "B-explicit": 1, "I-explicit": 2}  # Only three labels for BIO
      return [label_mapping.get(tag, 0) for tag in tags]  # Use get to handle unknown tags, defaulting to 'O'


    reviews_df["bio_tags"] = reviews_df["text"].apply(label_bio_tags_dynamic_v2) # Apply spaCy processing on CPU
    reviews_df = cudf.from_pandas(reviews_df) # Convert to cudf after spaCy processing
    dataset = Dataset.from_pandas(reviews_df.to_pandas()).shuffle(seed=42) # Convert back to pandas for Dataset

    total_size = len(dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.15 * total_size)

    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, train_size + val_size))
    test_data = dataset.select(range(train_size + val_size, total_size))

    return train_data, val_data, test_data

def create_dataloader(dataset, tokenizer, batch_size=4):
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=164, return_tensors="pt")

        labels = []
        for i, bio_tags in enumerate(examples["bio_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned_labels = [-100] * len(tokenized['input_ids'][i])
            for j, word_id in enumerate(word_ids):
                if word_id is not None and word_id < len(bio_tags):
                    aligned_labels[j] = bio_tags[word_id]
            labels.append(aligned_labels)

        tokenized["labels"] = labels
        return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    return DataLoader(tokenized_datasets, batch_size=batch_size)

# Model Evaluation
def evaluate_model(model, data_loader, device, tokenizer, task="evaluation"):
    model.eval()
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {task}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if isinstance(model, MultiTaskModel):
                _, token_classification_logits, _ = model(input_ids, attention_mask)
            elif isinstance(model, BiLSTMCRF):
                token_classification_logits = model(input_ids)
            else:
                _, token_classification_logits = model(input_ids, attention_mask)

            predictions = torch.argmax(token_classification_logits, dim=-1).cpu().numpy()
            true_labels = labels.cpu().numpy()

            all_predictions.extend(predictions.flatten())
            all_labels.extend(true_labels.flatten())

    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    accuracy = accuracy_score(all_labels, all_predictions)

    print(f"{task} - F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")

    token_accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"Token-Level Accuracy: {token_accuracy:.4f}")

    def compute_aspect_accuracy(predictions, labels, label_type):
        indices = np.where(np.array(labels) == label_type)[0]
        correct = np.sum(np.array(predictions)[indices] == np.array(labels)[indices])
        total = len(indices)
        return correct / total if total > 0 else 0.0

    explicit_accuracy = compute_aspect_accuracy(all_predictions, all_labels, 1)
    implicit_accuracy = compute_aspect_accuracy(all_predictions, all_labels, 2)
    print(f"Explicit Aspect Accuracy: {explicit_accuracy:.4f}")
    print(f"Implicit Aspect Accuracy: {implicit_accuracy:.4f}")

    def compute_aspect_coverage(predictions, labels, label_type):
        actual = np.sum(np.array(labels) == label_type)
        correctly_identified = np.sum((np.array(predictions) == label_type) & (np.array(labels) == label_type))
        return correctly_identified / actual if actual > 0 else 0.0

    explicit_coverage = compute_aspect_coverage(all_predictions, all_labels, 1)
    implicit_coverage = compute_aspect_coverage(all_predictions, all_labels, 2)
    print(f"Explicit Aspect Coverage: {explicit_coverage:.4f}")
    print(f"Implicit Aspect Coverage: {implicit_coverage:.4f}")

    return f1, precision, recall, accuracy

# Training Loop
# Training Loop
def train_model(model, train_loader, val_loader, device, tokenizer, max_epochs=3, gradient_accumulation_steps=2):
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)  # Reduced learning rate
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * max_epochs)
    model.to(device)

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if isinstance(model, MultiTaskModel):
                loss, _, _ = model(input_ids, attention_mask, token_classification_labels=labels)
            elif isinstance(model, (BERTBIOTagger, StandaloneMLM)):
                loss, _ = model(input_ids, attention_mask, labels=labels)
            elif isinstance(model, BiLSTMCRF):
                logits = model(input_ids)
                loss = nn.CrossEntropyLoss()(logits.view(-1, 3), labels.view(-1))
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

        evaluate_model(model, val_loader, device, tokenizer, task="validation")
#funbction to plot attention heatmap for above 20 short reviews  
def visualize_attention_heatmap(model, sentence, tokenizer, device):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=164).to(device)
    with torch.no_grad():
        outputs = model.shared_encoder.bert(**inputs, output_attentions=True)

    attention = outputs.attentions[-1].mean(dim=1).squeeze(0).cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Find the indices of [CLS] and [SEP] tokens
    cls_index = tokens.index('[CLS]')
    sep_index = tokens.index('[SEP]')

    # Remove [CLS] and [SEP] tokens and corresponding attention weights
    tokens = [token for i, token in enumerate(tokens) if i != cls_index and i != sep_index]
    attention = np.delete(attention, cls_index, axis=0)
    attention = np.delete(attention, cls_index, axis=1)

    # Update sep_index after removing [CLS]
    sep_index -= 1  # Adjust sep_index as it shifts by one after deleting [CLS]

    attention = np.delete(attention, sep_index, axis=0) # Now use updated sep_index
    attention = np.delete(attention, sep_index, axis=1)


    num_tokens = len(tokens)

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention[:num_tokens, :num_tokens], annot=True, fmt=".2f", xticklabels=tokens[:num_tokens], yticklabels=tokens[:num_tokens], cmap="viridis")
    plt.title("Attention Weights Heatmap")
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.xticks(rotation=90)
    plt.show()
  # Plot timestamp to capture computation time of each model
def get_timestamp():
  return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# Main Workflow
file_path = "/content/Gift_Cards.jsonl"
train_data, val_data, test_data = load_and_preprocess_data(file_path)
train_loader = create_dataloader(train_data, tokenizer)
val_loader = create_dataloader(val_data, tokenizer)
test_loader = create_dataloader(test_data, tokenizer)

# Initialize Models
multi_task_model = MultiTaskModel()
bert_bio_tagger = BERTBIOTagger()
standalone_mlm = StandaloneMLM()
bilstm_crf = BiLSTMCRF(vocab_size=tokenizer.vocab_size)

# Train and Evaluate Multi-Task Model
print(f"Code execution started at: {get_timestamp()}")
print("\nTraining Multi-Task Model")
train_model(multi_task_model, train_loader, val_loader, device, tokenizer)
print("\nEvaluating Multi-Task Model on Test Set")
evaluate_model(multi_task_model, test_loader, device, tokenizer, task="multi-task test")
print(f"Code execution completed at: {get_timestamp()}")

# Evaluate Comparison Models
for model, name in zip([bert_bio_tagger, standalone_mlm, bilstm_crf], ["BIO Tagger", "Standalone MLM", "BiLSTM-CRF"]):
    print(f"\nTraining {name}")
    print(f"Code execution started at: {get_timestamp()}")
    train_model(model, train_loader, val_loader, device, tokenizer)
    print(f"\nEvaluating {name} on Test Set")
    evaluate_model(model, test_loader, device, tokenizer, task=f"{name} test")
    print(f"Code execution completed at: {get_timestamp()}")

df=pd.read_json(file_path)
short_sentences = []
for index, row in df.iterrows():
    sentence = row['text']
    if len(sentence.split()) < 30:
        short_sentences.append(sentence)
    if len(short_sentences) == 30:
        break

# Convert to a set to remove duplicates (maintaining order)
unique_short_sentences = list(dict.fromkeys(short_sentences))

# If you still have less than 20 unique sentences after removing duplicates
while len(unique_short_sentences) < 30:
    print("Not enough unique sentences. Add more sentences to the dataframe 'df' or lower the length requirement.")
    break

# Print or use the unique short sentences
for sentence in unique_short_sentences:
  print(sentence)

# Now lets visualise the attention heatmap for above sentences to understan predictions
for i in unique_short_sentences:
    sentence = i
    print(f"Visualizing attention for sentence: {sentence}")
    visualize_attention_heatmap(multi_task_model, sentence, tokenizer, device)  
