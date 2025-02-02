!pip install datasets
!pip install spacy
!python -m spacy download en_core_web_sm
import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizerFast,  # Import BertTokenizerFast
    BertForTokenClassification,
    BertForMaskedLM,
    AdamW,
    get_scheduler
)
from datasets import Dataset

# Load Tokenizer using BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Multi-Task Model
# Multi-Task Learning Model (α = 0.7 fixed)
class MultiTaskModel(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", alpha=0.7):
        super().__init__()
        self.alpha = alpha  # Task Balance Hyperparameter
        self.shared_encoder = BertForTokenClassification.from_pretrained(pretrained_model_name)
        self.token_classification_head = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, 3))
        self.mlm_head = BertForMaskedLM.from_pretrained(pretrained_model_name).cls

    def forward(self, input_ids, attention_mask, token_classification_labels=None, mlm_labels=None):
        shared_output = self.shared_encoder.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        token_classification_logits = self.token_classification_head(shared_output.last_hidden_state)
        mlm_logits = self.mlm_head(shared_output.last_hidden_state)

        loss = 0
        if token_classification_labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            token_classification_loss = loss_fn(token_classification_logits.view(-1, 3), token_classification_labels.view(-1))
            loss += self.alpha * token_classification_loss

        if mlm_labels is not None:
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(mlm_logits.view(-1, tokenizer.vocab_size), mlm_labels.view(-1))
            loss += (1 - self.alpha) * mlm_loss

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
def load_and_preprocess_data(file_path, num_rows=500000,chunksize=10000):
    # reviews_df_chunks = pd.read_json(file_path, lines=True, chunksize=chunksize)
    # reviews_df = pd.concat([chunk for chunk in reviews_df_chunks])
    reviews_df=df
    reviews_df['text']=reviews_df['Sentence']
    reviews_df = reviews_df.head(num_rows)
    
    reviews_df = reviews_df.drop_duplicates(subset="text").reset_index(drop=True)

    def label_bio_tags_dynamic(text):
      doc = nlp(text)
      tokens = [token.text for token in doc]
      tags = ["O"] * len(tokens)

      for i, token in enumerate(doc):
          if token.pos_ in {"NOUN", "ADJ"} and (token.dep_ in {"amod", "nsubj", "dobj"}):
              tags[i] = "B" if tags[i] == "O" else "I"
          if token.pos_ in {"AUX", "VERB"} and token.dep_ == "ROOT" and "neg" in [child.dep_ for child in token.children]:
              for child in token.children:
                  if child.dep_ in {"attr", "acomp", "xcomp", "pobj"}:
                      child_index = tokens.index(child.text)
                      tags[child_index] = "B" if tags[child_index] == "O" else "I"

      label_mapping = {"O": 0, "B": 1, "I": 2}
      return [label_mapping[tag] for tag in tags]

    reviews_df["bio_tags"] = reviews_df["text"].apply(label_bio_tags_dynamic)
    dataset = Dataset.from_pandas(reviews_df).shuffle(seed=42)

    total_size = len(dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.15 * total_size)

    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, train_size + val_size))
    test_data = dataset.select(range(train_size + val_size, total_size))

    return train_data, val_data, test_data

def create_dataloader(dataset, tokenizer, batch_size=8):
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

        labels = []
        for i, bio_tags in enumerate(examples["bio_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned_labels = [-100] * len(tokenized['input_ids'][i])
            for j, word_id in enumerate(word_ids):
                if word_id is not None and word_id < len(bio_tags):
                    aligned_labels[j] = bio_tags[word_id]
            labels.append(aligned_labels)

        tokenized["labels"] = labels
        
        # Add MLM labels (Mask 15% of tokens)
        # Create mlm_labels, initially a copy of input_ids
        tokenized["mlm_labels"] = tokenized["input_ids"].clone()
        # Create a mask with 15% probability of True (masking)
        mask_indices = np.random.rand(*tokenized["input_ids"].shape) < 0.15  
        # Set mlm_labels to -100 where mask is False (not masking)
        tokenized["mlm_labels"][~mask_indices] = -100  
        
        return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Include 'mlm_labels' in the output format
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels', 'mlm_labels'])  
    return DataLoader(tokenized_datasets, batch_size=batch_size)

# Evaluation Function for Explicit Aspects (BIO Tagging)
def evaluate_explicit_aspect(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Explicit Aspects"):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

            loss, token_classification_logits, _ = model(input_ids, attention_mask, token_classification_labels=labels)

            predictions = torch.argmax(token_classification_logits, dim=-1).cpu().numpy()
            true_labels = labels.cpu().numpy()

            all_preds.extend(predictions.flatten())
            all_labels.extend(true_labels.flatten())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)

    print("\n🔹 Explicit Aspect Evaluation Metrics:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return f1, precision, recall, accuracy

# Evaluation Function for Implicit Aspects (MLM-based)
def evaluate_implicit_aspect(model, data_loader):
    model.eval()
    total_correct, total_predictions = 0, 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Implicit Aspects (MLM)"):
            input_ids, attention_mask, mlm_labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["mlm_labels"].to(device)

            _, _, mlm_logits = model(input_ids, attention_mask, mlm_labels=mlm_labels)
            predictions = torch.argmax(mlm_logits, dim=-1)

            mask_indices = (mlm_labels != -100)
            correct = (predictions[mask_indices] == mlm_labels[mask_indices]).sum().item()
            total = mask_indices.sum().item()

            total_correct += correct
            total_predictions += total

    implicit_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
    print(f"\n🔹 Implicit Aspect (MLM) Accuracy: {implicit_accuracy:.4f}")
    return implicit_accuracy

# Combined Evaluation Function
def evaluate_model_complete(model, test_loader):
    print("\n🔹 Starting Complete Model Evaluation...")
    
    # Evaluate Explicit Aspect Identification
    explicit_f1, explicit_precision, explicit_recall, explicit_accuracy = evaluate_explicit_aspect(model, test_loader)
    
    # Evaluate Implicit Aspect Identification
    implicit_accuracy = evaluate_implicit_aspect(model, test_loader)

    # Print Final Results
    print("\n🔹 Final Evaluation Summary:")
    print(f"Explicit Aspect Accuracy: {explicit_accuracy:.4f}")
    print(f"Explicit Aspect F1-Score: {explicit_f1:.4f}")
    print(f"Implicit Aspect Accuracy (MLM-based): {implicit_accuracy:.4f}")

    return {
        "explicit_f1": explicit_f1,
        "explicit_precision": explicit_precision,
        "explicit_recall": explicit_recall,
        "explicit_accuracy": explicit_accuracy,
        "implicit_accuracy": implicit_accuracy
    }



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
def train_model(model, train_loader, val_loader, device, tokenizer, max_epochs=5):
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)  # Reduced learning rate
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * max_epochs)
    model.to(device)

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

        evaluate_model(model, val_loader, device, tokenizer, task="validation")

# Main Workflow
file_path = "/content/Laptop_Train_v2.csv"
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
print("\nTraining Multi-Task Model")
train_model(multi_task_model, train_loader, val_loader, device, tokenizer)
print("\nEvaluating Multi-Task Model on Test Set")
# Run Evaluation on Test Data
evaluation_results = evaluate_model_complete(multi_task_model, test_loader)
evaluate_model(multi_task_model, test_loader, device, tokenizer, task="multi-task test")
evaluation_results
# Evaluate Comparison Models
for model, name in zip([bert_bio_tagger, standalone_mlm, bilstm_crf], ["BIO Tagger", "Standalone MLM", "BiLSTM-CRF"]):
    print(f"\nTraining {name}")
    train_model(model, train_loader, val_loader, device, tokenizer)
    print(f"\nEvaluating {name} on Test Set")
    evaluate_model(model, test_loader, device, tokenizer, task=f"{name} test")
