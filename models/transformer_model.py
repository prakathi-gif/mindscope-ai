"""
Model 3 & 4: Deep Transformer NLP Framework (RoBERTa and MentalBERT).
Supports fine-tuning, validation checkpointing, embedding extraction, and multi-class inference on CPU/GPU.
"""

import time
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModel, AutoConfig
)
from models.evaluator import evaluate_classification_predictions
from config import (
    ROBERTA_CHECKPOINT, MENTAL_ROBERTA_CHECKPOINT,
    PATH_ROBERTA_WEIGHTS, LABEL_NAMES, NUM_CLASSES
)

class MentalHealthDataset(Dataset):
    """PyTorch Dataset for text classification."""
    def __init__(self, texts: List[str], labels: Optional[np.ndarray], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class TransformerClassifier:
    """End-to-End Fine-Tuning & Inference Engine for RoBERTa and MentalBERT."""

    def __init__(
        self,
        checkpoint: str = ROBERTA_CHECKPOINT,
        num_classes: int = NUM_CLASSES,
        max_length: int = 128,
        device: Optional[str] = None
    ):
        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.config = AutoConfig.from_pretrained(checkpoint, num_labels=num_classes)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, config=self.config
        ).to(self.device)
        self.is_trained = False
        self.train_history: List[Dict[str, float]] = []

    def train_epoch(self, dataloader: DataLoader, optimizer, scheduler, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item()
        return total_loss / max(1, len(dataloader))

    def evaluate_loss(self, dataloader: DataLoader, criterion) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / max(1, len(dataloader)), (correct / max(1, total))

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: List[str],
        val_labels: np.ndarray,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 2
    ):
        """Fine-tunes the transformer with validation checkpointing."""
        train_dataset = MentalHealthDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = MentalHealthDataset(val_texts, val_labels, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.train_history = []
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer, None, criterion)
            val_loss, val_acc = self.evaluate_loss(val_loader, criterion)
            
            history_entry = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc)
            }
            self.train_history.append(history_entry)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.is_trained = True
        return self

    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Computes softmax probability distributions."""
        self.model.eval()
        dataset = MentalHealthDataset(texts, None, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        probs_list = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                probs_list.append(probs)

        return np.vstack(probs_list)

    def extract_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Extracts mean-pooled sentence embeddings from penultimate transformer layer."""
        self.model.eval()
        dataset = MentalHealthDataset(texts, None, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Use base model (e.g. roberta)
                base = getattr(self.model, "roberta", None) or getattr(self.model, "bert", None)
                if base is not None:
                    outputs = base(input_ids=input_ids, attention_mask=attention_mask)
                    # Mean pooling over attention mask
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    mean_pooled = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                embeddings.append(mean_pooled)

        return np.vstack(embeddings)

    def evaluate(self, test_texts: List[str], test_labels: np.ndarray, model_label: str = "Fine-Tuned RoBERTa") -> Dict[str, Any]:
        probs = self.predict_proba(test_texts)
        res = evaluate_classification_predictions(
            test_labels, probs, class_names=LABEL_NAMES, model_name=model_label
        )
        res["train_history"] = self.train_history
        return res

    def save(self, filepath=PATH_ROBERTA_WEIGHTS):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath=PATH_ROBERTA_WEIGHTS):
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.is_trained = True
        return self

