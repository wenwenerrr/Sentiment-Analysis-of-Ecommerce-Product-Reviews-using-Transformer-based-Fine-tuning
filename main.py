#!/usr/bin/env python3
"""
main.py

Single-entry script to run:
  1) Full fine-tuning (DistilBERT) with best hyperparameters
  2) LoRA fine-tuning with chosen LoRA hyperparameters

Outputs:
  - Summary table (printed) with Performance (F1/precision/recall/accuracy), Eval Loss
  - Training runtime (seconds) for each run
  - Peak GPU memory (MB) during training (if CUDA available)
  - Saves results to ./results/results_summary.csv

Notes:
  - Data CSV must be uploaded to /content/Womens Clothing E-Commerce Reviews.csv
  - Dependencies: transformers, datasets, peft, torch, scikit-learn, pandas, matplotlib
"""

import os
import time
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from torch import nn
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# PEFT / LoRA
from peft import LoraConfig, get_peft_model, TaskType

# -------------------------
# Helper functions
# -------------------------
def combine_text(row):
    title = str(row["Title"]).strip()
    review = str(row["Review Text"]).strip()
    return f"{title}. {review}" if title else review

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_metrics_from_preds(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return accuracy, precision, recall, f1

def print_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# -------------------------
# Device utilities
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

def get_peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    else:
        return 0.0

# -------------------------
# Configs: best hyperparams
# -------------------------
# Full fine-tuning best params (from your grid)
FULL_HP = {
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "dropout": 0.3,
    "num_epochs": 5,
    "batch_size": 16
}

# LoRA best params (fixed as requested)
LORA_HP = {
    "r": 4,
    "alpha": 16,
    "lora_dropout": 0.1,
    "learning_rate": 1e-3,
    "num_epochs": 4,
    "batch_size": 32
}

# Paths
CSV_PATH = "/content/Womens Clothing E-Commerce Reviews.csv"
RESULTS_DIR = "./results"
ensure_dir(RESULTS_DIR)

# -------------------------
# Data loading & preprocessing (shared)
# -------------------------
print("Loading dataset from:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# Keep only required columns and drop NA
df = df[["Title", "Review Text", "Recommended IND"]].dropna()

# Build input text
df["text"] = df.apply(combine_text, axis=1)
df["clean_text"] = df["text"].apply(clean_text)

print("Total available samples:", len(df))
print("Label distribution:")
print(df["Recommended IND"].value_counts(normalize=True))

# -------------------------
# Tokenizer (shared)
# -------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# -------------------------
# Metric function for Trainer compatibility (not used for printing table)
# -------------------------
def trainer_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc, prec, rec, f1 = compute_metrics_from_preds(preds, labels)
    return {
        "eval_accuracy": acc,
        "eval_precision": prec,
        "eval_recall": rec,
        "eval_f1": f1
    }

# -------------------------
# Weighted Trainer classes (compatible with newer transformers)
# -------------------------
class WeightedTrainer(Trainer):
    """Trainer that uses class weights in CrossEntropyLoss.
       Accepts extra kwargs (e.g., num_items_in_batch) from Trainer internal calls.
    """
    def __init__(self, class_weights_tensor, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_tensor = class_weights_tensor
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor, label_smoothing=self.label_smoothing)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

class WeightedLoRATrainer(Trainer):
    """Trainer for LoRA model that uses class weights."""
    def __init__(self, class_weights_tensor, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_tensor = class_weights_tensor
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor, label_smoothing=self.label_smoothing)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# -------------------------
# Train Full Fine-tuning
# -------------------------
def train_full_finetune(df, hp=FULL_HP):
    """Train DistilBERT via full fine-tuning using the provided hyperparameters.
       Returns: dict with metrics, runtime (s), peak GPU mem (MB), and confusion matrix.
    """
    print("\n--- Running Full Fine-tuning ---")
    # follow your earlier logic: sample up to 10000 for experiment if dataset large
    sample_n = min(len(df), 10000)
    df_small = df.sample(sample_n, random_state=42)

    train_df, test_df = train_test_split(
        df_small,
        test_size=0.2,
        random_state=42,
        stratify=df_small["Recommended IND"]
    )

    # make datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # tokenize
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    train_dataset = train_dataset.rename_column("Recommended IND", "labels")
    test_dataset = test_dataset.rename_column("Recommended IND", "labels")

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # class weights
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["Recommended IND"]
    )
    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float).to(device)
    print("Class weights:", class_weights_arr)

    # model init with dropout as requested
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        dropout=hp["dropout"],
        attention_dropout=hp["dropout"]
    )
    model.to(device)

    # training args
    training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "full_ft"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=hp["num_epochs"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        logging_dir="./logs_full",
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        save_total_limit=1
    )

    trainer = WeightedTrainer(
        class_weights_tensor=class_weights_tensor,
        label_smoothing=0.0,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=trainer_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # reset GPU mem counters
    reset_gpu_memory_stats()
    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time
    peak_mem_mb = get_peak_memory_mb()
    # evaluation predictions
    eval_out = trainer.predict(test_dataset)
    logits = eval_out.predictions
    preds = np.argmax(logits, axis=1)
    labels = eval_out.label_ids
    acc, prec, rec, f1 = compute_metrics_from_preds(preds, labels)
    eval_loss = float(eval_out.metrics.get("test_loss", np.nan))

    cm = print_confusion(labels, preds)

    # save model checkpoint (optional: commented out since user only wants outputs)
    # model.save_pretrained("./best_finetuned_model")
    # tokenizer.save_pretrained("./best_finetuned_model")

    result = {
        "model": "full_finetune",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "eval_loss": eval_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "train_time_s": duration,
        "peak_gpu_mem_mb": peak_mem_mb
    }
    return result

# -------------------------
# Train LoRA (parameter-efficient)
# -------------------------
def train_lora(df, hp=LORA_HP):
    """Train DistilBERT with LoRA adapters using the provided hyperparameters.
       Returns: dict with metrics, runtime (s), peak GPU mem (MB), and confusion matrix.
    """
    print("\n--- Running LoRA Fine-tuning ---")
    # follow your earlier logic: sample up to 8000 for experiment (as in your LoRA script)
    sample_n = min(len(df), 8000)
    df_small = df.sample(sample_n, random_state=42)

    train_df, test_df = train_test_split(
        df_small,
        test_size=0.2,
        random_state=42,
        stratify=df_small["Recommended IND"]
    )

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # tokenize
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    train_dataset = train_dataset.rename_column("Recommended IND", "labels")
    test_dataset = test_dataset.rename_column("Recommended IND", "labels")

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # class weights
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["Recommended IND"]
    )
    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float).to(device)
    print("Class weights:", class_weights_arr)

    # base model
    base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # LoRA config (use specified best parameters)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=hp["r"],
        lora_alpha=hp["alpha"],
        lora_dropout=hp["lora_dropout"],
        target_modules=["q_lin", "v_lin"],
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(base_model, lora_config)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100.0 * trainable_params / all_params
    print(f"Trainable params: {trainable_params:,} ({trainable_percentage:.4f}%)")

    # training args
    training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "lora"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=hp["num_epochs"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir="./logs_lora",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        save_total_limit=1
    )

    trainer = WeightedLoRATrainer(
        class_weights_tensor=class_weights_tensor,
        label_smoothing=0.0,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=trainer_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # measure time and peak memory
    reset_gpu_memory_stats()
    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time
    peak_mem_mb = get_peak_memory_mb()

    # evaluation
    eval_out = trainer.predict(test_dataset)
    preds = np.argmax(eval_out.predictions, axis=1)
    labels = eval_out.label_ids
    acc, prec, rec, f1 = compute_metrics_from_preds(preds, labels)
    eval_loss = float(eval_out.metrics.get("test_loss", np.nan))
    cm = print_confusion(labels, preds)

    # do not save model by default (user requested only outputs)
    result = {
        "model": "lora",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "eval_loss": eval_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "train_time_s": duration,
        "peak_gpu_mem_mb": peak_mem_mb,
        "trainable_percentage": trainable_percentage
    }
    return result

# -------------------------
# Main: run both experiments and print table
# -------------------------
def main():
    print("\nStarting experiments: Full Fine-tuning and LoRA (one run each)\n")

    results_list = []

    # Full fine-tune
    full_res = train_full_finetune(df, hp=FULL_HP)
    results_list.append(full_res)

    # LoRA
    lora_res = train_lora(df, hp=LORA_HP)
    results_list.append(lora_res)

    # Build summary dataframe
    rows = []
    for r in results_list:
        rows.append({
            "model": r["model"],
            "train_samples": r["train_samples"],
            "test_samples": r["test_samples"],
            "eval_loss": r["eval_loss"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "train_time_s": r["train_time_s"],
            "peak_gpu_mem_mb": r["peak_gpu_mem_mb"]
        })

    summary_df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)

    # Print a nice table
    pd.set_option("display.precision", 4)
    print("\n\n===== Final Comparison =====")
    print(summary_df.to_string(index=False))

    # Save results
    summary_csv = os.path.join(RESULTS_DIR, "results_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to {summary_csv}")

    # Also show confusion matrices (print)
    print("\nConfusion matrices:")
    for r in results_list:
        print(f"\nModel: {r['model']}")
        print(r["confusion_matrix"])

if __name__ == "__main__":
    main()
