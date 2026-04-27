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
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Reverse mapping for label names
label_names = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}

# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Visualization functions
def plot_confusion_matrix(y_true, y_pred, output_path="./confusion_matrix.png"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[label_names[i] for i in range(3)],
        yticklabels=[label_names[i] for i in range(3)],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Model Predictions', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")
    plt.close()

def plot_classification_metrics(y_true, y_pred, output_path="./classification_metrics.png"):
    """Plot precision, recall, and F1-score per class"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2]
    )
    
    x = np.arange(len(label_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#2E86AB')
    ax.bar(x, recall, width, label='Recall', color='#A23B72')
    ax.bar(x + width, f1, width, label='F1-Score', color='#F18F01')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Metrics by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label_names[i] for i in range(3)])
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Classification metrics saved to: {output_path}")
    plt.close()

def plot_prediction_distribution(y_pred, output_path="./prediction_distribution.png"):
    """Plot distribution of predictions"""
    unique, counts = np.unique(y_pred, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#E63946', '#F77F00', '#06A77D']
    bars = ax.bar(
        [label_names[i] for i in unique],
        counts,
        color=colors,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Model Predictions', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Prediction distribution saved to: {output_path}")
    plt.close()

def evaluate_model(model, tokenizer, test_df, output_dir="./evaluation_results"):
    """Evaluate model on test set and generate visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("EVALUATING MODEL")
    print("="*50)
    
    # Prepare test data
    test_texts = test_df['News_Title'].tolist()
    test_labels = test_df['label'].tolist()
    
    # Tokenize
    encodings = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='weighted'
    )
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(
        test_labels,
        predictions,
        target_names=[label_names[i] for i in range(3)],
        digits=4
    ))
    
    # Generate visualizations
    plot_confusion_matrix(
        test_labels,
        predictions,
        f"{output_dir}/confusion_matrix.png"
    )
    plot_classification_metrics(
        test_labels,
        predictions,
        f"{output_dir}/classification_metrics.png"
    )
    plot_prediction_distribution(
        predictions,
        f"{output_dir}/prediction_distribution.png"
    )
    
    print(f"\nAll evaluation results saved to: {output_dir}")
    print("="*50 + "\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
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
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        fp16=torch.cuda.is_available() if torch.cuda.is_available() else False
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

    # Evaluate model and generate visualizations
    evaluate_model(model, tokenizer, test_df)

else:
    print("Please upload your CSV file first!")