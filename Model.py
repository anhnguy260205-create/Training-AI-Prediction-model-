import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
import os

# Label mapping (3 classes)
label_map = {
    'Bullish': 2,
    'Somewhat-Bullish': 2,
    'Positive': 2,

    'Neutral': 1,

    'Somewhat-Bearish': 0,
    'Bearish': 0,
    'Negative': 0
}

file_path = "sentiment_training_pool.csv"

if os.path.exists(file_path):
    print("Loading dataset...")

    df = pd.read_csv(file_path)

    # Keep only valid labels
    df = df[df['Official_Label'].isin(label_map.keys())].copy()

    # Create numeric labels
    df['label'] = df['Official_Label'].map(label_map)

    # Keep only required columns
    df = df[['News_Title', 'label']].dropna()

    # Train-test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # Load FinBERT tokenizer + model
    model_name = "ProsusAI/finbert"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # IMPORTANT
    )

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    def tokenize_func(examples):
        return tokenizer(
            examples["News_Title"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_train = train_dataset.map(tokenize_func, batched=True)
    tokenized_test = test_dataset.map(tokenize_func, batched=True)

    # Training settings
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    print("Start training...")
    trainer.train()

    # Save model
    save_path = "./my_finbert_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("Training finished!")
    print(f"Model saved to: {save_path}")

else:
    print("Please upload your CSV file first!")