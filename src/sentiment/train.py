"""Fine-tune DistilBERT for 3-class sentiment classification.

Tuned for CPU-only training: short max sequence length (these are short
tweets), small batch size, limited epochs. Uses class weighting to correct
for the significant imbalance in the mapped sentiment labels (Neutral is
~15x smaller than Negative/Frustrated after emotion->sentiment mapping).
"""

import logging

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from configs.config import MODELS_DIR
from src.sentiment.preprocessing import (
    ID_TO_SENTIMENT,
    SENTIMENT_LABELS,
    load_emotion_dataset,
)
from src.utils.device import get_device

logger = logging.getLogger(__name__)

SENTIMENT_MODEL_DIR = MODELS_DIR / "sentiment"
BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64  # short tweets — no need for longer sequences


class SentimentDataset(Dataset):
    """Wraps tokenized text + labels for the Trainer API."""

    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies class weights to the loss function.

    Standard cross-entropy treats every class equally, which would let the
    model largely ignore the small Neutral class while still achieving
    decent overall accuracy. Class weighting penalizes mistakes on
    under-represented classes more heavily, which is necessary here given
    Neutral is ~15x smaller than Negative/Frustrated.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def train_sentiment_classifier() -> None:
    device = get_device()
    logger.info("Training on device: %s", device)

    train_df, val_df, _ = load_emotion_dataset()

    logger.info("Loading tokenizer and base model: %s", BASE_MODEL_NAME)
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=len(SENTIMENT_LABELS)
    )
    model.to(device)

    train_dataset = SentimentDataset(train_df["text"], train_df["sentiment_id"], tokenizer)
    val_dataset = SentimentDataset(val_df["text"], val_df["sentiment_id"], tokenizer)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(SENTIMENT_LABELS)),
        y=train_df["sentiment_id"],
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info(
        "Class weights (%s): %s",
        [ID_TO_SENTIMENT[i] for i in range(len(SENTIMENT_LABELS))],
        class_weights_tensor.tolist(),
    )

    training_args = TrainingArguments(
        output_dir=str(SENTIMENT_MODEL_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        use_cpu=(device == "cpu"),
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )

    logger.info("Starting training (this will take a while on CPU)...")
    trainer.train()

    eval_results = trainer.evaluate()
    logger.info("Final validation results: %s", eval_results)

    SENTIMENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SENTIMENT_MODEL_DIR / "final")
    tokenizer.save_pretrained(SENTIMENT_MODEL_DIR / "final")
    logger.info("Saved fine-tuned model and tokenizer to %s", SENTIMENT_MODEL_DIR / "final")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_sentiment_classifier()